import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow import keras
import mysql.connector

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'


# Load the pre-trained pothole detection model
pothole_model = keras.models.load_model('pothole.h5')

# Function to process and classify images
def classify_pothole(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (150, 150))
    img_resized = img_resized / 255.0

    # Make a prediction
    prediction = pothole_model.predict(np.expand_dims(img_resized, axis=0))
    is_pothole = prediction[0][0] > 0.5  # Adjust the threshold as needed

    return is_pothole


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Route for the upload page
@app.route('/')
def upload_page():
    return render_template('upload.html')

# Route to handle the uploaded file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        is_pothole = classify_pothole(filename)
     
        if is_pothole:
            result = 'Pothole Detected'
        else:
            result = 'No Pothole Detected'

        return result

if __name__ == '__main__':
    app.run(debug=True)