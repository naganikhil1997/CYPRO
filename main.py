from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import os

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
origins = ["http://localhost", "http://localhost:3000","https://cypro-2.onrender.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
MODEL_PATH = r"C:\Users\prudh\OneDrive\Desktop\dd1\models\best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
MODEL = YOLO(MODEL_PATH)

# Class names from the model
CLASS_NAMES = ['Bacterial', 'Downy mildew', 'Healthy', 'Powdery mildew', 'Septoria Blight', 'Virus', 'Wilt - Leaf Blight']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file MIME type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, JPEG, and PNG are allowed.")

    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Run inference
    results = MODEL.predict(source=image, conf=0.25)

    # Draw bounding boxes if detections exist
    draw = ImageDraw.Draw(image)
    labels = []
    confidences = []

    # Increase font size and change color to neon yellowish green
    try:
        font = ImageFont.truetype("arial.ttf", 30)  # Increased font size
    except IOError:
        font = ImageFont.load_default()

    neon_yellow_green = (57, 255, 20)  # Neon yellowish green color

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = CLASS_NAMES[int(box.cls.item())]
            confidence = float(box.conf.item())

            labels.append(label)
            confidences.append(confidence)

            # Draw rectangle with neon yellowish green
            draw.rectangle([x1, y1, x2, y2], outline=neon_yellow_green, width=5)

            # Create text inside the box with neon yellowish green
            text = f"{label} ({confidence:.2f})"
            text_bbox = draw.textbbox((x1, y1), text, font=font)  # Get text size
            text_x1, text_y1, text_x2, text_y2 = text_bbox

            # Draw filled rectangle behind text for visibility
            draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill=neon_yellow_green)

            # Draw text inside the bounding box in black for contrast
            draw.text((text_x1, text_y1), text, fill="black", font=font)

    # Convert the annotated image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Return annotated image as a response
    return Response(content=img_bytes.getvalue(), media_type="image/png")