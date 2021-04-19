from keras.models import model_from_json,load_model
from flask import Flask,request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

model = model_from_json(open("AI Models/Nik_model.json", "r").read())
model.load_weights('AI Models/Nik_model.h5')

@app.route('/')
def index():
    return "<h1> Deployed to Heroku</h1>"

@app.route('/predict',methods=['POST'])
def predict():

    img = request.files['image']
    img.save('Static/image.jpg')
    img2=cv2.imread('Static/image.jpg')
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    facecasc = cv2.CascadeClassifier('Static/haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        #print(emotion_dict[maxindex])

    return jsonify({'emotion_detected':emotion_dict[maxindex]})



if '__name__'=='__main__':

    #app.run()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

