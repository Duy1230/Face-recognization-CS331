import cv2
import numpy as np
import torchvision.transforms as transforms


preprocess = transforms.Compose([
    transforms.ToPILImage(),  # Convert from numpy array to PIL image
    transforms.Resize((160, 160)),  # Resize to the required input size of InceptionResnetV1
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

def calculate_overlap(box1, box2):
    """
    Calculate the overlap between two bounding boxes in the YOLO convention.
    
    Args:
        box1 (list): [x1, y1, x2, y2] coordinates of the first bounding box.
        box2 (list): [x1, y1, x2, y2] coordinates of the second bounding box.
        
    Returns:
        float: The overlap ratio between the two bounding boxes.
    """
    # Convert the bounding boxes to NumPy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Calculate the coordinates of the intersection rectangle
    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])
    
    # Compute the area of intersection
    intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    
    # Compute the area of union
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = box1_area + box2_area - intersection
    
    # Compute the overlap ratio
    overlap_ratio = intersection / union
    
    return overlap_ratio
def chooseColor(isBlink, isFakeBlink):
    if isBlink:
        return (255, 0, 0)
    else:
        return (0, 0, 255)
    
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = cv2.resize(frame, (224, 224))  # Resize frame to match VGGFace2 input size
    frame = frame.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    frame = transforms.ToTensor()(frame)  # Convert frame to tensor
    frame = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(frame)  # Normalize frame
    return frame.unsqueeze(0)  # Add batch dimension

def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))