import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image

# Ensure Keras backend consistency
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if not needed

app = Flask(__name__)

# Print TensorFlow version (optional for debugging)
print(f"TensorFlow version: {tf.__version__}")

# Load the pre-trained model
# Load the pre-trained model
MODEL_PATH = 'model.h5'  # Make sure this path is correct
model = None  # Initialize model variable

# Debugging: Print the current working directory and list files
print(f"Current working directory: {os.getcwd()}")
print(f"Files in the current directory: {os.listdir(os.getcwd())}")

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' does not exist.")
else:
    try:
        model = load_model(MODEL_PATH)
        print('Model loaded successfully.')
    except OSError as e:
        print(f"OSError: {e} - Check if the file path is correct and the file is not corrupted.")
    except ValueError as e:
        print(f"ValueError: {e} - There might be a version compatibility issue.")
    except Exception as e:
        print(f"Unknown error occurred while loading the model: {e}")



# Define your class labels
LABELS = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Prediction function
def get_prediction(image_path):
    if model is None:
        return "Error: Model is not loaded."
    
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(225, 225))
        x = img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict the class probabilities
        predictions = model.predict(x)
        if predictions.size == 0:
            return "Error: No prediction returned."
        
        # Get the label with the highest probability
        predicted_label = LABELS[np.argmax(predictions[0])]
        return predicted_label
    except Exception as e:
        return f"Prediction error: {e}"

# Flask routes
@app.route('/', methods=['GET'])
def index():
    # Display the upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in the request."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    # Secure and save the file
    try:
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(upload_folder, secure_filename(file.filename))
        file.save(file_path)

        # Make a prediction
        prediction = get_prediction(file_path)
        return prediction
    except Exception as e:
        return f"File processing error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
