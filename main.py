import cv2
import fastapi
import numpy as np
import uvicorn
import joblib
from PIL import Image
from fastapi import File, UploadFile
from skimage.feature import hog, local_binary_pattern

app = fastapi.FastAPI()

xgb = joblib.load("xgb_model_71.70.pkl")
R = 1
P = 8
grid_size = (2, 2)
n_bins = P*2 + 2

label_list = ['ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades',
 'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades',
 'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
 'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades',
 'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
 'joker', 'king of clubs', 'king of diamonds', 'king of hearts',
 'king of spades', 'nine of clubs', 'nine of diamonds', 'nine of hearts',
 'nine of spades', 'queen of clubs', 'queen of diamonds', 'queen of hearts',
 'queen of spades', 'seven of clubs', 'seven of diamonds', 'seven of hearts',
 'seven of spades', 'six of clubs', 'six of diamonds', 'six of hearts',
 'six of spades', 'ten of clubs', 'ten of diamonds', 'ten of hearts',
 'ten of spades', 'three of clubs', 'three of diamonds', 'three of hearts',
 'three of spades', 'two of clubs', 'two of diamonds', 'two of hearts',
 'two of spades']

def preprocess_image(img):
    img = img.resize((224, 224))
    img = img.convert("L")

    #detect edges
    img = np.array(img)
    img = cv2.Canny(img, 100, 200)

    # extract features
    hog_features = hog(img,orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    lbp = local_binary_pattern(img, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins))
    feature_vector = np.append(hog_features, hist)

    return feature_vector

@app.post("/classify-card")
def index(image: UploadFile = File(...)):
    img = Image.open(image.file)
    feature_vector = preprocess_image(img)
    feature_vector = feature_vector.reshape(1, -1)
    prediction = xgb.predict(feature_vector)
    # decode the prediction
    if(prediction[0] < 0 or prediction[0] > 53):
        label = "unknown"
    else:
        label = label_list[prediction[0]]
    return {"prediction": label, "confidence": float(xgb.predict_proba(feature_vector).max()), "label_length": len(label_list)}
