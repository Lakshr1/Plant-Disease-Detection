# Plant Disease Detection

This repository contains a plant disease detection application that leverages machine learning techniques to identify diseases in crop leaves. The application is hosted at [https://plant-disease-detection-lakshr1.streamlit.app/](https://plant-disease-detection-lakshr1.streamlit.app/).

## Publication Status

This project is currently under the publication process. Stay tuned for updates!

## About Dataset

The dataset used for this project is recreated using offline augmentation from the original dataset, which can be found on [Keggle on this link]([https://github.com/username/original-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)). It consists of approximately 87,000 RGB images of healthy and diseased crop leaves categorized into 38 different classes. The dataset is divided into the following directories:

- **train**: Contains 70,295 images.
- **test**: Contains 33 images for prediction purposes.
- **validation**: Contains 17,572 images.

## Model

The model used for plant disease detection is MobileNet, a lightweight convolutional neural network architecture designed for mobile and embedded vision applications. MobileNet balances between model size and accuracy, making it suitable for deployment in resource-constrained environments.

To handle class imbalance in the dataset, the Synthetic Minority Over-sampling Technique (SMOTE) oversampling technique is employed. SMOTE generates synthetic samples for the minority classes, equalizing the representation of all 38 categories in the dataset during training.

## Model Performance

The trained model achieved an accuracy of 97.4%, outperforming models that do not utilize SMOTE oversampling, which typically achieve an accuracy of 94.1%. The higher accuracy indicates the effectiveness of SMOTE in addressing class imbalance.

## Challenges Encountered

One of the main challenges encountered during model training was overfitting. Overfitting occurs when the model learns to memorize the training data instead of generalizing well to unseen data. To mitigate overfitting, the following techniques were applied:

- **Data Augmentation**: Increasing the diversity of training examples through techniques like rotation, flipping, and scaling.
- **Dropout**: Randomly dropping neurons during training to prevent co-adaptation of features.
- **Regularization**: Adding penalties on model parameters to prevent them from becoming too large.



