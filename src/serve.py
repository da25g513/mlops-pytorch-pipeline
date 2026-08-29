import io
import os
from pathlib import Path
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
from src.model import get_model

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/checkpoints/classifier_v1.pt",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ]
)

app = FastAPI(
    title="CIFAR-10 Model API",
    version="1.0.0",
)

model = None

def load_model():
    global model
    checkpoint_path = Path(MODEL_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )
    return {
        "status": "ok",
        "model_loaded": True,
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )
    try:
        image_bytes = await image.read()
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        tensor = transform(pil_image)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[0]

        predicted_index = int(
            probabilities.argmax()
        )

        probability_map = {
            CLASS_NAMES[index]: round(
                float(probabilities[index]),
                6,
            )
            for index in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "probabilities": probability_map,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc