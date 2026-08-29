import torch
from src.model import get_model

def test_model_output_shape():
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )
    model.eval()
    inputs = torch.randn(
        2,
        3,
        32,
        32,
    )
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == (2, 10)

def test_model_has_ten_classes():
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )
    assert model.fc.out_features == 10