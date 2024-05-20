

import cv2
import numpy as np

def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):

    if quantized:
        if modelFile == None:
            modelFile = "models/opencv_face_detector_uint8.pb"                       #opencv face detector model
        if configFile == None:
            configFile = "models/\.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        
    else:
        if modelFile == None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"            # caffe model
        if configFile == None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):      # This Function basically use for  finding List of coordinates of the faces detected in the image
    """
    Find the faces in an image
    
    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    model : dnn_Net
        Face detection model

    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image

    """


    h, w = img.shape[:2]                                                 # grab frame dimension
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,       #convert it into the blob
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]       #finding the confidence
        if confidence > 0.5:                  #compute x and y of the bounding box for the object
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces                                         # List of coordinates of the faces detected in the image

def draw_faces(img, faces):       # it will create rectangle along the face
    """
    Draw faces on image

    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
    faces : List of face coordinates
        Coordinates of faces to draw

    Returns
    -------
    None.

    """
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)


        