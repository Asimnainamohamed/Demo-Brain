import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading and Class Labels ---
# IMPORTANT: This code assumes your trained model file, 'model.h5', is in the same directory as this app.py file.
# Make sure to upload your 'model.h5' file to Render along with this script.
try:
    model_path = os.path.join(os.getcwd(), 'model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None to handle errors in the prediction route

# Define the class labels based on your notebook
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# --- Helper Functions ---

def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image bytes for model prediction.
    Resizes the image to 150x150 pixels and converts it to a numpy array.
    
    Args:
        image_bytes: Bytes of the uploaded image file.
        
    Returns:
        A preprocessed numpy array or None if the image is invalid.
    """
    try:
        # Open the image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Check if the image has a single channel (grayscale)
        # If it's a valid MRI image, it should have a single channel.
        if img.mode != 'L':  # 'L' mode is for single-channel grayscale images
            print("Invalid image mode. Not a grayscale image.")
            return None

        # Convert to RGB to ensure compatibility, then resize and normalize
        img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        
        return img_array

    except (IOError, ValueError, TypeError) as e:
        print(f"Error processing image: {e}")
        return None

# --- API Routes ---

@app.route('/')
def home():
    """
    A simple home route to check if the API is running.
    """
    return "Brain Tumor Classification API is running. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the image prediction request.
    
    Expects a multipart/form-data POST request with an 'image' file.
    """
    # Check if a model was loaded successfully
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure 'model.h5' is present."}), 500

    # Check if a file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "Empty file name."}), 400

    try:
        # Read the image file bytes
        image_bytes = file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(image_bytes)
        
        # Handle cases where preprocessing fails (e.g., invalid image type)
        if processed_image is None:
            return jsonify({
                "error": "This is not a correct report.",
                "message": "The uploaded image is not a valid MRI scan or is in an unsupported format."
            }), 400
        
        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_labels[predicted_class_index]
        
        return jsonify({
            "prediction": predicted_class_name,
            "confidence": float(prediction[0][predicted_class_index])
        })
        
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # You can change the host and port for local testing
    # In production, Render will handle these.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
