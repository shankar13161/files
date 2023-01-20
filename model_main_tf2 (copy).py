import tensorflow as tf
import numpy as np
import cv2

labels_list = ['Backpack', 'Gitar', 'Cricket bat', 'Hockey stick']
from keras.models import load_model

#model = load_model('objects_detection_cnn_model_1.h5')
model = load_model('objects_detection_cnn_model_2.h5')

import tensorflow as tf
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    frame=cv2.flip(frame, 1)

    #define region for detection and prediction
    detection_box = frame[100:400, 320:620]
    detection_box = cv2.cvtColor(detection_box, cv2.COLOR_BGR2GRAY
                                )
    detection_box = cv2.resize(detection_box, (128, 128), interpolation = cv2.INTER_AREA)
    
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
    
    detected = detection_box.reshape(1, 128, 128, 1) 
    detected = detected / 255
    prediction = model.predict(detected)
    confidence = prediction[0][np.argmax(prediction)]

    cv2.putText(copy, labels_list[np.argmax(prediction)], (300 , 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    if confidence > 0.5:
        cv2.putText(copy, 'prediction: ' + str(confidence), (250 , 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(copy, 'prediction: ' + str(confidence), (250 , 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('frame', copy)    
    
    if cv2.waitKey(1) ==13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()