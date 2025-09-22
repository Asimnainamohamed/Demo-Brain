import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class labels (for demonstration)
class_labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    uploaded_image_path = None
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # If user doesn't select file, browser also submits an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # For demonstration: simulate a prediction
                # In a real app, this would use the model to make a prediction
                import random
                predicted_class = random.choice(class_labels)
                confidence = random.uniform(85.0, 99.9)
                
                prediction_result = {
                    'class': predicted_class.replace('_', ' ').title(),
                    'confidence': confidence
                }
                
                # Save the path for displaying the image
                uploaded_image_path = filename
            except Exception as e:
                return render_template('index.html', error=f'Error processing image: {str(e)}')
    
    return render_template('index.html', prediction=prediction_result, image=uploaded_image_path)

if __name__ == '__main__':
    app.run(debug=True)