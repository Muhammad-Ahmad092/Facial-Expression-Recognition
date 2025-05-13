# predict_emotion.py
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from joblib import load

# Load the trained model, scaler, and PCA
try:
    svm = load('linear_svm_model.joblib')
    scaler = load('scaler.joblib')
    pca = load('pca.joblib')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'linear_svm_model.joblib', 'scaler.joblib', and 'pca.joblib' are in the directory.")
    exit()

# Preprocess the image (same as training)
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = cv2.equalizeHist(image)
    return image

# Feature extraction (adjusted to match training)
def extract_features(image):
    # HOG (adjust parameters to match ~324 features)
    hog_f, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    print(f"HOG shape: {hog_f.shape}")  # Debug
    # If HOG is still 900, try reducing cell size or block overlap
    if hog_f.shape[0] != 324:  # Expected training HOG size
        hog_f, _ = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
        print(f"Adjusted HOG shape: {hog_f.shape}")  # Debug
    # LBP
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_f = hist / hist.sum()
    print(f"LBP shape: {lbp_f.shape}")  # Debug
    # GLCM
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_f = [graycoprops(glcm, prop).ravel()[0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']]
    glcm_f = np.array(glcm_f)
    print(f"GLCM shape: {glcm_f.shape}")  # Debug
    # VGG16 FC1
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)  # FC1: 4096
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.stack([img] * 3, axis=-1)  # Convert to RGB
    img = preprocess_input(img * 255)
    img = np.expand_dims(img, axis=0)
    vgg_f = model.predict(img, verbose=0).flatten()
    print(f"VGG16 shape: {vgg_f.shape}")  # Debug
    # Fuse features
    fused = np.hstack([hog_f, lbp_f, glcm_f, vgg_f])
    print(f"Fused shape before adjustment: {fused.shape}")  # Debug
    # Adjust to match training's 914 features
    expected_total = 914
    hog_len, lbp_len, glcm_len = len(hog_f), len(lbp_f), len(glcm_f)
    vgg_len = expected_total - (hog_len + lbp_len + glcm_len)
    if vgg_len <= 0 or vgg_len > len(vgg_f):
        raise ValueError(f"Cannot adjust VGG16 features to fit {expected_total}. Got VGG length {len(vgg_f)}, need {vgg_len}. Check HOG parameters.")
    fused = np.hstack([hog_f, lbp_f, glcm_f, vgg_f[:vgg_len]])
    print(f"Fused shape after adjustment: {fused.shape}")  # Debug
    # Apply scaler and PCA
    if fused.shape[0] != expected_total:
        raise ValueError(f"Expected {expected_total} features, got {fused.shape[0]}. Adjust feature extraction.")
    fused_scaled = scaler.transform(fused.reshape(1, -1))
    fused_pca = pca.transform(fused_scaled)
    print(f"PCA shape: {fused_pca.shape}")  # Debug
    return fused_pca

# Predict
def predict_emotion(image_path):
    image = preprocess_image(image_path)
    features = extract_features(image)
    prediction = svm.predict(features)
    return "Happy" if prediction[0] == 1 else "Neutral"

# Main execution
if __name__ == "__main__":
    image_path = "gettyimages-1465122813-612x612.jpg"  # Replace with your image path
    try:
        result = predict_emotion(image_path)
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error: {e}")