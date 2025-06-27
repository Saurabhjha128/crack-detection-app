from flask import Flask, request, render_template, jsonify, url_for
print("Starting app.py...")
from flask_sqlalchemy import SQLAlchemy
print("Imported Flask-SQLAlchemy successfully.")
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime, timezone
print("Imported all other dependencies successfully.")

app = Flask(__name__, 
            template_folder="../frontend/templates", 
            static_folder="../frontend/static")
print("Flask app initialized.")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
print("SQLAlchemy initialized.")

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bounding_box = db.Column(db.String(200))
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
print("Prediction model defined.")

with app.app_context():
    db.create_all()
print("Database created.")

device = torch.device("cpu")
print(f"Using device: {device}")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
print("Model architecture loaded.")

model_path = "crack_model_final.pth"
print(f"Model path: {model_path}")
print(f"Model path exists: {os.path.exists(model_path)}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")
model.load_state_dict(torch.load(model_path, map_location=device))
print("Model weights loaded successfully.")

model.eval()
model = model.to(device)
print("Model set to evaluation mode and moved to device.")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Transform pipeline defined.")

@app.route('/')
def index():
    print("Accessed index route.")
    print("Attempting to render index.html...")
    try:
        result = render_template('index.html')
        print("Rendered index.html successfully.")
        print(f"Rendered content: {result[:100]}...")  # Print first 100 chars of rendered content
        return result
    except Exception as e:
        print(f"Error rendering index.html: {e}")
        return "Error rendering template", 500

@app.route('/predict', methods=['POST'])
def predict():
    print("Accessed predict route.")
    try:
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        img_array = np.array(img)

        uploads_dir = os.path.join(app.static_folder, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join('uploads', secure_filename(file.filename))
        full_image_path = os.path.join(app.static_folder, image_path)
        img.save(full_image_path)
        print(f"Saved uploaded image to {full_image_path}.")

        img_tensor = transform(img).unsqueeze(0).to(device)
        print("Image transformed and moved to device.")

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            label = "Crack" if prediction.item() == 1 else "No Crack"
            confidence = confidence.item()
        print(f"Prediction made: {label} (Confidence: {confidence})")

        bounding_box = {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150} if prediction.item() == 1 else None
        if prediction.item() == 1:
            img_array = cv2.rectangle(img_array, (50, 50), (150, 150), (0, 255, 0), 2)
            output_path = os.path.join(app.static_folder, 'output.jpg')
            print(f"Attempting to save output to: {output_path}")
            try:
                cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                print(f"Saved output image to {output_path}. File size: {os.path.getsize(output_path)} bytes.")
                print(f"File exists: {os.path.exists(output_path)}")
            except Exception as e:
                print(f"Error saving output image: {e}")
                return jsonify({'error': f"Failed to save output image: {e}"}), 500
        else:
            output_path = None

        pred = Prediction(
            image_path=image_path,
            label=label,
            confidence=confidence,
            bounding_box=json.dumps(bounding_box) if bounding_box else None,
            upload_date=datetime.now(timezone.utc)
        )
        db.session.add(pred)
        db.session.commit()
        print("Prediction saved to database.")

        return jsonify({
            'label': label,
            'confidence': round(confidence, 2),
            'image_url': url_for('static', filename='output.jpg') if output_path else None
        })
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    print("Accessed history route.")
    predictions = Prediction.query.order_by(Prediction.upload_date.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    print("Accessed clear_history route.")
    try:
        with app.app_context():
            db.session.query(Prediction).delete()
            db.session.commit()
        print("Prediction history cleared successfully.")
        return jsonify({'message': 'Prediction history cleared successfully.'})
    except Exception as e:
        print(f"Error clearing history: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=8080, debug=True)