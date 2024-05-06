import cv2
import torch
import numpy as np
import pathlib
from helper_functions import *
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
import os

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

DATABASE_PATH = 'database/data.json'
FACE_PATH = 'database/faces/'

# Initialize MTCNN and InceptionResnetV1 models
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
model_face = torch.hub.load('ultralytics/yolov5', 'custom', path='faceNew.pt', force_reload=False)
model_eye = torch.hub.load('ultralytics/yolov5', 'custom', path='eye.pt', force_reload=False)
model_eye.conf = 0.6

FACE_SIMILARITY_THRESHOLD = 0.9
OVERLAP_THRESHOLD = 0.93

def detect_for_frame(frame, face, is_blink, fake_blink, eye_state, prev_frame_face, prev_face):

    #detect face for frame
    results_face = model_face(frame)

    # Get the bounding box coordinates and labels
    bboxes = results_face.xyxy[0].cpu().numpy()
    labels = results_face.names

    if bboxes.size != 0:
        # only get bounding box with biggest area
        if bboxes.size > 1:
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            bboxes = bboxes[area.argmax()]
        
        #get current face bounding box
        current_frame_face = bboxes[:4]

        x1, y1, x2, y2 = [int(val) for val in bboxes[:4]]

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), chooseColor(is_blink, fake_blink), 2)

        temp_face = frame[y1:y2, x1:x2]
        face = temp_face.copy()
        
        #detect eye in the box region
        results_eye = model_eye(temp_face)

        # Get the bounding box coordinates and labels
        bboxes = results_eye.xyxy[0].cpu().numpy()
        state = [bbox[5] for bbox in bboxes]
        labels = results_eye.names
        
        if is_blink and not fake_blink:
            return frame, face[2:-2, 2:-2], is_blink, fake_blink, eye_state, prev_frame_face, prev_face

        
        offsets = [x1, y1, x1, y1]

        #if there are two eyes and both are the same state
        if len(bboxes) == 2 and state[0] == state[1]:
            

            for bbox in bboxes:
                x1, y1, x2, y2 = [(int(val) + offset) for val, offset in zip(bbox[:4], offsets)]
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw the label
                label_text = f"{labels[int(bbox[5])]} {bbox[4]:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                
                #if that eye is a new state
                if (bbox[5] not in eye_state):
                    eye_state.append(bbox[5])

                #if all state happened
            if len(eye_state) == 2:
                    #current_face_area = (current_frame_face[2] - current_frame_face[0]) * (current_frame_face[3] - current_frame_face[1])
                    #prev_face_area = (prev_frame_face[2] - prev_frame_face[0]) * (prev_frame_face[3] - prev_frame_face[1])
                    
                if not is_blink:
                    
                    if calculate_overlap(current_frame_face, prev_frame_face) > OVERLAP_THRESHOLD and prev_face is not None:
                        face_similarity = isTheSameFace(face, prev_face)
                        if face_similarity > FACE_SIMILARITY_THRESHOLD:
                            cv2.imwrite("assets/d1.png", face)
                            cv2.imwrite("assets/d2.png", prev_face)
                            is_blink = True
                            print(face_similarity)
                        else:
                            prev_face = None
                            print(f"Nice try but {face_similarity} is not similar to previous face")
                            eye_state = []
                        #fake_blink = False
                    else:
                        eye_state = []
                        fake_blink = True
        else:
            eye_state = []

        prev_frame_face = current_frame_face
        prev_face = face

        # if is_blink and not fake_blink:
        #     cv2.putText(frame, "blink", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        # if fake_blink and not is_blink:
        #     cv2.putText(frame, "fake blink", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    else:
        eye_state = []
        face = None
        prev_face = None


    if face is not None:
        return frame, face[2:-2, 2:-2], is_blink, fake_blink, eye_state, prev_frame_face, prev_face
    else:
        return frame, face, is_blink, fake_blink, eye_state, prev_frame_face, prev_face


def extractFeature(face):
    img_tensor = preprocess(face)
    embeddings = resnet(img_tensor.unsqueeze(0)).detach()
    return embeddings

def addFaceToDatabase(face, name):

    #Add face image to image database
    cv2.imwrite(os.path.join(FACE_PATH, name+'.png'), face)

    #Open json database
    with open(DATABASE_PATH, 'r') as file:
        data = json.load(file)

    #Add face data to json database
    new_data = {
        "id": data["numUser"],
        "name": name,
        "feature": extractFeature(face).tolist()
    }

    #Update json database
    data["numUser"] += 1
    data["features"].append(new_data)

    #Write json database
    with open(DATABASE_PATH, 'w') as file:
        json.dump(data, file, indent=4)

def faceMatching(face):
    embeddings = extractFeature(face)

    similarity = []
    read_names = []
    read_features = []
    with open(DATABASE_PATH, 'r') as file:
        data = json.load(file)
        for user in data["features"]:
            read_features.append(np.array(user["feature"]))
            read_names.append(user["name"])

    for i in range(len(read_features)):
        similarity.append(cosine_similarity(embeddings.squeeze(), read_features[i].squeeze()))
    print(read_names)
    print(similarity)
    max_index = similarity.index(max(similarity))
    return read_names[max_index], max(similarity)

def isTheSameFace(face1, face2):
    feature_1 = extractFeature(face1)
    feature_2 = extractFeature(face2)
    return cosine_similarity(feature_1.squeeze(), feature_2.squeeze())