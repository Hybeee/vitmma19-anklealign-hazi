import torch
from torchvision import transforms
import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
import uvicorn
from contextlib import asynccontextmanager

from config import Args
from models import load_trained_model_from_path
from data_pipeline.data_preparing import _resize_with_padding as resize_with_padding

MODEL = None
ARGS = None
DEVICE = None

def get_latest_model_path(base_dir='outputs'):
    if not os.path.exists(base_dir):
        return None
    
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not subdirs:
        return None
    
    subdirs.sort(reverse=True)

    for folder in subdirs:
        model_path = os.path.join(folder, "model.pth")
        if os.path.exists(model_path):
            return model_path
        
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, ARGS, DEVICE

    print("Loading Configuration")
    ARGS = Args()
    DEVICE = ARGS.device

    print("Finding latest model...")
    model_path = get_latest_model_path(ARGS.output_dir)
    if not model_path:
        raise RuntimeError(f"Could not find a trained model in {ARGS.output_dir}")
    
    print(f"Loading Model from: {model_path}")
    MODEL = load_trained_model_from_path(model_path=model_path)
    MODEL.to(DEVICE)
    MODEL.eval()
    print("Model loaded and ready for inference!")

    yield

    print("Cleaning up resources...")
    del MODEL
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="AnkleAlign Inference API", lifespan=lifespan)

def preprocess_image(args: Args, image):
    if args.use_padding:
        image = resize_with_padding(args, image)
    else:
        image = cv2.resize(image, (args.resolution, args.resolution))
    
    image = np.array(image, dtype=np.float32)
    image /= 255.0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    ])
    
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_bytes =np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        input_tensor = preprocess_image(args=ARGS, image=image)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}
    
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)

        outputs = MODEL(input_tensor)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_index = torch.max(outputs, 1)

        predicted_index = predicted_index.item()
        predicted_class = ARGS.classes[predicted_index]
        confidence = probabilities[0][predicted_index].item() * 100

        probs_dict = {ARGS.classes[i]: f"{probabilities[0][i].item() * 100}" for i in range(len(ARGS.classes))}

        return {
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}",
            "class_probabilities": probs_dict
        }
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)