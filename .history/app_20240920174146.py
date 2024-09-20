from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("model/best.pt")  # Make sure the path to your model is correct

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Get the uploaded image
    image_file = request.files['image'].read()
    image = Image.open(io.BytesIO(image_file))

    # Resize image to 64x64 (the size the model was trained on)
    image = image.resize((64, 64))

    # Convert the image to a tensor for YOLO model
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform prediction
    results = model.predict(image_tensor)

    # Get the top 5 predictions
    top5 = results[0].probs.top5
    top5_conf = results[0].probs.top5conf

    # Get the class names
    class_names = model.names
    predictions = [{"class": class_names[top5[i]], "confidence": float(top5_conf[i])} for i in range(5)]

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
