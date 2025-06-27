from flask import Flask, request, render_template, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os
import json
from datetime import datetime, timezone

# Initialize Flask app with frontend folders
app = Flask(__name__, 
            template_folder="../frontend/templates", 
            static_folder="../frontend/static")

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define prediction result model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bounding_box = db.Column(db.String(200))
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Create DB
with app.app_context():
    db.create_all()

# Load model
device = torch.device("cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model_path = os.path.join(os.path.dirname(__file__), "crack_model_final.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Define transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- ROUTES ---------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        img_array = np.array(img)

        uploads_dir = os.path.join(app.static_folder, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join('uploads', secure_filename(file.filename))
        full_image_path = os.path.join(app.static_folder, image_path)
        img.save(full_image_path)

        # Prepare image for model
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            label = "Crack" if prediction.item() == 1 else "No Crack"
            confidence = confidence.item()

        # Dummy bounding box for "Crack"
        bounding_box = {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150} if label == "Crack" else None
        output_path = None

        if label == "Crack":
            # Draw bounding box
            img_array = cv2.rectangle(img_array, (50, 50), (150, 150), (0, 255, 0), 2)
            output_path = os.path.join(app.static_folder, 'output.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        # Save to DB
        pred = Prediction(
            image_path=image_path,
            label=label,
            confidence=confidence,
            bounding_box=json.dumps(bounding_box) if bounding_box else None,
            upload_date=datetime.now(timezone.utc)
        )
        db.session.add(pred)
        db.session.commit()

        return jsonify({
            'label': label,
            'confidence': round(confidence, 2),
            'image_url': url_for('static', filename='output.jpg') if output_path else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    predictions = Prediction.query.order_by(Prediction.upload_date.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        with app.app_context():
            db.session.query(Prediction).delete()
            db.session.commit()
        return jsonify({'message': 'Prediction history cleared successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
