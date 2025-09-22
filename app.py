# app.py
import io
import os
import json
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# ---------- Configuration ----------
MODEL_PATH = os.environ.get("MODEL_PATH", "effnet.h5")  # place your model here
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.json")  # optional mapping
IMG_SIZE = (224, 224)  # change to your model's expected input size
CONFIDENCE_THRESHOLD = 0.60  # below this -> "can't predict"
EDGE_DENSITY_THRESHOLD = 0.01  # too low edge density -> likely unrelated
FOREGROUND_RATIO_THRESHOLD = 0.02  # too little foreground -> likely unrelated
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}

# ---------- App init ----------
app = Flask(__name__)

# Load model (Keras .h5 or SavedModel)
if not os.path.exists(MODEL_PATH):
    app.logger.warning(f"Model file not found at {MODEL_PATH}. Start-up will still work, but /predict will fail until model is provided.")
    model = None
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        model = None

# Load labels if provided
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        try:
            labels = json.load(f)
        except Exception:
            labels = None
else:
    labels = None

# If no labels file, create a placeholder (change as appropriate)
# e.g. labels = { "0": "no_tumor", "1": "glioma", ... }
if labels is None and model is not None:
    # attempt to create numeric labels
    try:
        out_dim = model.output_shape[-1]
        labels = {str(i): str(i) for i in range(out_dim)}
    except Exception:
        labels = { "0": "class_0" }

# ---------- Utilities ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def read_image_from_filestorage(file_storage):
    contents = file_storage.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(img)

def preprocess_image(img_array, img_size=IMG_SIZE):
    # img_array: HxWx3 uint8
    img = cv2.resize(img_array, img_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # expand dims for batch
    return np.expand_dims(img, axis=0)

def check_unrelated_image(img_array):
    """
    Heuristic checks to decide whether the image is 'unrelated' (not a brain MRI)
    Returns (is_related_bool, reasons_list)
    """
    reasons = []
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Foreground ratio: proportion of non-white/black pixels
    # Use Otsu threshold to separate foreground
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = np.count_nonzero(th == 0) / (h * w)  # black region ratio
    if fg_ratio < FOREGROUND_RATIO_THRESHOLD:
        reasons.append(f"low_foreground_ratio={fg_ratio:.4f}")

    # Edge density: amount of edges (Canny)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w)
    if edge_density < EDGE_DENSITY_THRESHOLD:
        reasons.append(f"low_edge_density={edge_density:.4f}")

    # Brightness/contrast sanity (prevent all-white or all-black)
    mean_val = float(np.mean(gray)) / 255.0
    if mean_val < 0.02 or mean_val > 0.98:
        reasons.append(f"mean_intensity_out_of_range={mean_val:.3f}")

    is_related = len(reasons) == 0
    return is_related, reasons

def postprocess_predictions(preds):
    # preds: model output (batch x classes)
    if isinstance(preds, list):
        preds = np.array(preds)
    preds = np.squeeze(preds)
    if preds.ndim == 0:
        # single value (binary logit) -> convert to 2-class softmax-like
        prob = float(tf.sigmoid(preds).numpy())
        return {"0": 1.0 - prob, "1": prob}
    # if not already probabilities, try softmax
    if preds.sum() > 1.0001 or preds.sum() < 0.9999:
        probs = tf.nn.softmax(preds).numpy()
    else:
        probs = preds
    # Build label->prob dict
    result = {}
    for i, p in enumerate(probs):
        key = labels.get(str(i), str(i)) if labels else str(i)
        result[key] = float(p)
    return result

# ---------- Routes ----------
@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "Brain MRI classification API. POST an image to /predict (form field 'image')."
    })

@app.route("/predict", methods=["POST"])
def predict():
    # Basic checks
    if model is None:
        return jsonify({"error": "Model not loaded on server. Upload model.h5 to the server path."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image' in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File extension not allowed"}), 400

    try:
        img_array = read_image_from_filestorage(file)  # RGB numpy
    except Exception as e:
        return jsonify({"error": f"Unable to read image: {str(e)}"}), 400

    # Quick unrelated-image heuristic
    is_related, reasons = check_unrelated_image(img_array)
    if not is_related:
        return jsonify({
            "predicted": None,
            "confidence": None,
            "related": False,
            "message": "This does not look like a correct brain MRI / expected input. Cannot produce a reliable prediction.",
            "reasons": reasons
        }), 200

    # Preprocess & predict
    x = preprocess_image(img_array)
    try:
        preds = model.predict(x)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    probs = postprocess_predictions(preds)
    # Find top label
    top_label = max(probs, key=lambda k: probs[k])
    top_conf = probs[top_label]

    if top_conf < CONFIDENCE_THRESHOLD:
        return jsonify({
            "predicted": None,
            "confidence": float(top_conf),
            "related": True,
            "message": "Model confidence too low to provide a reliable report. Cannot produce a reliable prediction.",
            "probabilities": probs
        }), 200

    return jsonify({
        "predicted": top_label,
        "confidence": float(top_conf),
        "related": True,
        "message": "Prediction successful",
        "probabilities": probs
    }), 200

# ---------- Run (for local dev only) ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
