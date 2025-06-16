import torch
from models.cnn import resnet50
from utils.dataset import create_data_loader
from utils.train_eval import evaluate
from sklearn.metrics import classification_report
import argparse

# python test.py --test_dir ".\data\test" --model_path ".\best_model.pth"

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader, test_classes = create_data_loader(
        args.test_dir,
        args.batch_size,
        is_train=False
    )

    model = resnet50(num_classes=len(test_classes))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    loss, acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # --- raport klasyfikacji ---
    print("\nDetailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=test_classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test dataset")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to test data folder")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    args = parser.parse_args()
    main(args)
