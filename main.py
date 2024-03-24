# import streamlit as st

# st.write("hi stramlit")
# st.subheader("sub heading")
# st.selectbox("which lang you like",['python','java'])
# st.checkbox("Python")
# st.checkbox("Java")
# st.slider("PLease rate the project" ,0,100)
# st.select_slider("Please rate us",["best","avg","worst"])
# st.progress(10)


# st.sidebar.title("About")
# st.sidebar.selectbox("which lang you like",['python','java','C'])
# st.sidebar.markdown("information")
# st.sidebar.button("information")

import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):

    model = tf.keras.models.load_model('my_model_saved.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])





#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of a plant with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases.
    3. *Results:* View the results and recommendations for further action.

    ### Why Choose Us?
    - *Accuracy:* Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - *User-Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
    


elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    st.link_button("Download Test Images","https://github.com/Lakshr1/Plant-Disease-Detection/tree/main/test/test")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        predicted_class = class_name[result_index]
        st.success("Model is Predicting it's a {}".format(predicted_class))

        # Nutritional information
        nutrition_map = {
            'Apple___healthy': "High in fiber, vitamin C, and antioxidants. Low in calories and fat.",
            'Blueberry___healthy': "High in antioxidants, vitamin C, and dietary fiber.",
            'Cherry_(including_sour)___healthy': "High in antioxidants, vitamin C, and potassium.",
            'Corn_(maize)___healthy': "Rich in carbohydrates, fiber, and vitamins.",
            'Grape___healthy': "Rich in antioxidants, vitamin C, and fiber.",
            'Peach___healthy': "High in vitamin C, vitamin A, and dietary fiber.",
            'Pepper,_bell___healthy': "Low in calories, high in vitamin C and antioxidants.",
            'Potato___healthy': "High in carbohydrates, vitamin C, and potassium.",
            'Raspberry___healthy': "High in fiber, vitamin C, and antioxidants.",
            'Soybean___healthy': "High in protein, fiber, and various vitamins and minerals.",
            'Strawberry___healthy': "High in vitamin C, fiber, and antioxidants.",
            'Tomato___healthy': "High in vitamin C, vitamin A, and antioxidants."
        }

        # Diseases for which treatment involves fertilizer
        treatable_diseases = {
            'Apple___Apple_scab': ["Nitrogen", "Phosphorus", "Potassium"],
            'Apple___Black_rot': ["Nitrogen", "Calcium", "Magnesium"],
            'Apple___Cedar_apple_rust': ["Potassium", "Boron", "Copper"],
            'Grape___Black_rot': ["Nitrogen", "Calcium", "Magnesium"],
            'Grape___Esca_(Black_Measles)': ["Potassium", "Boron", "Copper"],
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Bacterial_spot': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Early_blight': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Late_blight': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Leaf_Mold': ["Potassium", "Boron", "Copper"],
            'Tomato___Septoria_leaf_spot': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Spider_mites Two-spotted_spider_mite': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Target_Spot': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ["Nitrogen", "Phosphorus", "Potassium"],
            'Tomato___Tomato_mosaic_virus': ["Nitrogen", "Phosphorus", "Potassium"]
        }

        # # Diseases for which there is no effective treatment
        # untreatable_diseases = {
        #     'Cherry_(including_sour)___Powdery_mildew': "Non-edible due to disease.",
        #     'Orange___Haunglongbing_(Citrus_greening)': "Non-edible due to disease.",
        #     'Peach___Bacterial_spot': "Non-edible due to disease.",
        #     # Add more untreatable diseases and their status
        # }

        if predicted_class in nutrition_map:
            st.write("Edible")
            st.write("Nutritional Information")
            st.write(nutrition_map[predicted_class])
        elif predicted_class in treatable_diseases:
            st.write("Edible")
            st.write("Disease Detected")
            st.write("Treatment with Fertilizer Required:")
            st.write("Fertilizer Recommendations:", treatable_diseases[predicted_class])
        else:
            st.warning("Non-edible: No specific treatment for complete recovery ")

