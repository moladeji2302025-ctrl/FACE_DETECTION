# app.py
import os
import io
import base64
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from db_init import db, Usage, app as db_app

# Use the same Flask app instance as db_init to share SQLAlchemy
app = db_app

# Configuration
MODEL_PATH = 'trained_models/empathica_v1.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load model
print("Loading model from:", MODEL_PATH)
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded.")
else:
    print("Warning: Model file not found. Train model with model.py and place at", MODEL_PATH)

# emotion labels (must match the order used during training)
# If you trained with ImageDataGenerator, check train_generator.class_indices from training to map indexes.
EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image: Image.Image):
    # Convert to grayscale 48x48, scale, return shape (1,48,48,1)
    image = image.convert('L').resize((48,48))
    arr = img_to_array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,h,w,1) if using grayscale channel last
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts either:
    - An uploaded image file under 'image'
    - A JSON payload with 'image_base64' = dataURL (from webcam capture on client)
    - Optional form field 'name' for DB
    Returns JSON with predicted emotion and confidence, and saves a record in DB.
    """
    if model is None:
        return jsonify({'error':'Model not loaded. Train model and put file at ' + MODEL_PATH}), 500

    name = request.form.get('name', None)
    # 1) File upload
    if 'image' in request.files and request.files['image'].filename != '':
        f = request.files['image']
        if not allowed_file(f.filename):
            return jsonify({'error':'Invalid file type'}), 400
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        save_name = f"{ts}_{f.filename}"
        save_path = os.path.join(UPLOAD_FOLDER, save_name)
        f.save(save_path)
        image = Image.open(save_path)
    else:
        # 2) Base64 in form field 'image_base64'
        data_url = request.form.get('image_base64', None)
        if not data_url:
            return jsonify({'error':'No image provided'}), 400
        # data_url looks like "data:image/png;base64,XXXXX"
        header, encoded = data_url.split(',',1)
        binary = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary))
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        ext = header.split('/')[1].split(';')[0]
        save_name = f"{ts}_webcam.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, save_name)
        image.save(save_path)

    # Preprocess and predict
    x = preprocess_image(image)  # shape (1,48,48,1)
    # If model expects shape (1,48,48,1) or (1,48,48,3) - make sure match training
    preds = model.predict(x)
    idx = int(np.argmax(preds))
    emotion = EMOTIONS[idx] if idx < len(EMOTIONS) else str(idx)
    confidence = float(preds[0][idx])

    # Save usage record
    record = Usage(name=name, image_path=save_path, predicted_emotion=emotion, confidence=confidence)
    db.session.add(record)
    db.session.commit()

    # Return result page or JSON
    if request.form.get('return') == 'html':
        return render_template('result.html', emotion=emotion, confidence=confidence, image_url=save_path)
    else:
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'image_path': save_path
        })

if __name__ == '__main__':
    # Ensure DB exists
    db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
