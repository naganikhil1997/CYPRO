import os
from ultralytics import YOLO

# MODEL_PATH = r"C:\Users\prudh\OneDrive\Desktop\dd1\models\best.pt"
MODEL_PATH = r".\models\best.pt"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load the YOLO model
MODEL = YOLO(MODEL_PATH)

print("Model loaded successfully!")
