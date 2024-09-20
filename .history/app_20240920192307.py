from flask import Flask, request, jsonify, send_file
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("model/best.pt")  # Make sure the path to your model is correct

# Root route
@app.route('/')
def home():
    return "Welcome to the Green Seaweed Flask App!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Get the uploaded image
    image_file = request.files['image'].read()
    image = Image.open(io.BytesIO(image_file))

    # Resize image to 64x64 (the size the model was trained on)
    image_resized = image.resize((64, 64))

    # Convert the image to a tensor for YOLO model
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform prediction
    results = model.predict(image_tensor)

    # Get the top 5 predictions
    top5 = results[0].probs.top5
    top5_conf = results[0].probs.top5conf

    # Get the class names
    class_names = model.names
    predictions = [{"class": class_names[top5[i]], "confidence": float(top5_conf[i])} for i in range(5)]

    # Annotate image with top 5 predictions
    annotated_image = np.array(image)  # Convert PIL image to OpenCV-compatible array
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR (OpenCV format)
    font_scale = max(annotated_image.shape) / 1000.0
    font_thickness = max(1, int(max(annotated_image.shape) / 500))

    highlight_colors = [(0, 255, 0),  # Green for the top prediction
                        (0, 255, 255),  # Yellow
                        (173, 216, 230),  # Light Blue
                        (0, 165, 255),  # Orange
                        (255, 0, 255)]  # Magenta

    y_offset = 30  # Start below the top of the image

    for i, prediction in enumerate(predictions):
        color = highlight_colors[i % len(highlight_colors)]  # Cycle through the colors
        label = f"{prediction['class']}: {prediction['confidence']:.2f}"

        # Put the text prediction on the image
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(annotated_image, (10, y_offset - text_height), (10 + text_width, y_offset + 10), color, cv2.FILLED)
        cv2.putText(annotated_image, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        y_offset += text_height + 20  # Move down for the next label

    # Resize the image for display (optional)
    max_width = 800
    max_height = 600
    h, w, _ = annotated_image.shape
    scale = min(max_width / w, max_height / h)
    resized_annotated_image = cv2.resize(annotated_image, (int(w * scale), int(h * scale)))

    # Convert back to RGB for saving/display
    annotated_image_rgb = cv2.cvtColor(resized_annotated_image, cv2.COLOR_BGR2RGB)
    annotated_pil_image = Image.fromarray(annotated_image_rgb)

    # Save the image to a BytesIO object
    image_io = io.BytesIO()
    annotated_pil_image.save(image_io, format="JPEG")
    image_io.seek(0)

    # Return predictions and the annotated image
    return send_file(image_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
