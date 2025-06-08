from flask import Flask, request, jsonify,send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins="*`")  # Allow all origins for development
# Load the model and labels
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Load disease info from JSON
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# Helper function to preprocess image
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

# Prediction function
def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

# API Endpoint for Prediction
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the Plant Disease Recognition API"}), 200



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']
    
    # Save the image temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
    temp_path = os.path.join('uploadimages', temp_filename)
    image.save(temp_path)
    
    try:
        # Get prediction
        prediction = model_predict(temp_path)
        
        # Return JSON response
        return jsonify({
            "success": True,
            "prediction": prediction,
            "image_url": f"/uploadimages/{temp_filename}"  # Optional: URL to access the image later
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up: Delete the temp file (optional)
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Serve uploaded images (optional)
@app.route('/uploadimages/<filename>')
def serve_image(filename):
    return send_from_directory('uploadimages', filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)