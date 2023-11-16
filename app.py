
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile

app = Flask(__name__)


model = load_model(r'my_model.h6') 
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive the image from the front-end
    uploaded_image = request.files['image']

    # Create a temporary file to save the uploaded image
    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, 'temp_image.jpg')
    uploaded_image.save(temp_image_path)

    # Load and preprocess the image
    img = image.load_img(temp_image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict the class probabilities
    probs = model.predict(img)[0]

    # Get the predicted class index and name
    pred_class_prob = np.argmax(probs)
    pred_class_name = class_names[pred_class_prob]

    # Format the result as a dictionary
    result = {
        # 'class_1': float(probs[0]),
        # 'class_2': float(probs[1]),
        'predicted_class': pred_class_name,
        'probability': float(probs[pred_class_prob])
    }

    # Clean up: Delete the temporary image file
    os.remove(temp_image_path)

    # Display the image with the predicted class and probability (optional)
    plt.figure(figsize=(15, 15))
    plt.imshow(img[0] / 255.)
    plt.axis('off')
    plt.text(10, 20, f'Predicted class: {pred_class_name}\nProbability: {probs[pred_class_prob]:.2f}', fontsize=20, color='red', bbox=dict(facecolor='white', alpha=0.8))
    plt.show()

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

