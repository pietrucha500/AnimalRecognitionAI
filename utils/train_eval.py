import os
import torch
import time

def save_checkpoint(model, optimizer, epoch, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    filename = f"epoch_{epoch}.pth"
    filepath = os.path.join(path, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint '{checkpoint_path}', resuming from epoch {start_epoch}")
    return model, optimizer, start_epoch

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(loader)
    print(f"\nTotal batches: {total_batches}")

    for i, (inputs, labels) in enumerate(loader):
        try:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                correct += (torch.eq(preds, labels)).sum().item()
            total += labels.size(0)

            del inputs, outputs, loss, preds, labels
            if i % 10 == 0:  # co 10 batchy
                torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                current_loss = running_loss / total
                current_acc = correct / total
                progress = (i + 1) / total_batches * 100
                print(f'\rProgress: [{i+1}/{total_batches}] {progress:.1f}% | Loss: {current_loss:.4f} | Acc: {current_acc:.4f}', end='')

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nERROR: Out of memory in batch: {i}. Skipping this batch.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(loader)
    print(f"\nValidation - Number of batches: {total_batches}")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (torch.eq(preds, labels)).sum().item()
            total += labels.size(0)

            del outputs, loss, preds
            torch.cuda.empty_cache()

            if (i + 1) % 5 == 0:
                current_loss = running_loss / total
                current_acc = correct / total
                progress = (i + 1) / total_batches * 100
                print(
                    f'\rValidation: [{i + 1}/{total_batches}] {progress:.1f}% | Loss: {current_loss:.4f} | Acc: {current_acc:.4f}',
                    end='')

    print()
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, checkpoint_path=None, save_freq=1, scaler=None, patience=5, min_delta=0.001):
    print("\nStarting training...")
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    best_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 1

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\nStarting epoch {epoch}/{num_epochs}")
            start = time.time()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("\nTraining phase...")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

            print("\nValidation phase...")
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            end = time.time()

            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
                  f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} - "
                  f"Time: {(end - start):.1f}s")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                print("Val loss improved! Saving model to best_model.pth!")
            else:
                patience_counter += 1
                print(f"No improvement since {patience_counter} epochs")

            if patience_counter >= patience:
                print(f"\nEarly stopping! No improvement since {patience} epochs.")
                print(f"Best val loss: {best_val_loss:.4f}")
                break

            if epoch % save_freq == 0:
                save_checkpoint(model, optimizer, epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print("Best model saved!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

    print(f"Best validation accuracy: {best_acc:.4f}")
    return best_val_loss, best_acc


