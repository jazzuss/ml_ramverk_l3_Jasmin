from dataset import get_dataloaders
from model import CIFAR10CNN
from train import train, get_device


def main() -> None:
    device = get_device()
    train_loader, test_loader = get_dataloaders(batch_size=64)
    model = CIFAR10CNN(dropout=0.25)

    train(
        model, train_loader, test_loader,
        lr=1e-3, epochs=10, device=device,
        save_path="models/best_model.pth",
    )


if __name__ == "__main__":
    main()