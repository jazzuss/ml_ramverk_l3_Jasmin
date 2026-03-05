import torch
from model import CIFAR10CNN

def export() -> None:
    model = CIFAR10CNN(dropout=0.25)
    model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        "models/model.onnx",
        input_names=["image"],
        output_names=["prediction"],
    )
    print("Exported to models/model.onnx")

if __name__ == "__main__":
    export()