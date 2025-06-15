import os
import argparse
import torch
from models.cnn import resnet50
from utils import train_eval
from utils.dataset import create_data_loader

# python main.py --train_dir ".\data\train" --val_dir ".\data\val" --epochs 1000 --batch_size 32

# od checkpointu:
# python main.py --train_dir ".\data\train" --val_dir ".\data\val" --checkpoint checkpoints/epoch_number.pth

def main(args):
    print(f"\nGPU Info:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2:.0f}MB")
        print(f"Free GPU Memory: {torch.cuda.memory_allocated(0) / 1024 ** 2:.0f}MB used")

    print("\nStarting program...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    print(f"Checking if directories exist:")
    print(f"Train dir: {args.train_dir} - exists: {os.path.exists(args.train_dir)}")
    print(f"Val dir: {args.val_dir} - exists: {os.path.exists(args.val_dir)}")

    print("\nLoading training data...")
    try:
        train_loader, train_classes = create_data_loader(
            args.train_dir,
            args.batch_size,
            is_train=True
        )
        print(f"Successfully loaded training data. Found {len(train_classes)} classes")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return

    print("\nLoading validation data...")
    try:
        val_loader, _ = create_data_loader(
            args.val_dir,
            args.batch_size,
            is_train=False
        )
        print("Successfully loaded validation data")
    except Exception as e:
        print(f"Error loading validation data: {str(e)}")
        return

    scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    model = resnet50(num_classes=len(train_classes))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_eval.train_model(model, train_loader, val_loader, criterion, optimizer, device,
                           num_epochs=args.epochs,
                           checkpoint_path=args.checkpoint,
                           save_freq=args.save_freq,
                           scaler=scaler,
                           patience=10,
                           min_delta=0.0005)

    model.load_state_dict(torch.load('best_model.pth'))
    val_loss, val_acc = train_eval.evaluate(model, val_loader, criterion, device)
    print(f"Final evaluation on validation set - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 on animal image classification dataset")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to training data folder")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to validation data folder")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument('--save_freq', type=int, default=1, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    main(args)