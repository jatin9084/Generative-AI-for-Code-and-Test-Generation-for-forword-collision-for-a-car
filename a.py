import cv2
import numpy as np

# Load pre-trained YOLO model (YOLOv3 or YOLOv4)
model_config = 'yolov4.cfg'
model_weights = 'yolov4.weights'
net = cv2.dnn.readNet(model_weights, model_config)

# Load COCO dataset class labels
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Function to detect objects
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)
    
    boxes, confidences, class_ids = [], [], []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'car':
                box = obj[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], confidences[i]) for i in indices.flatten()]

# Function to implement FCW logic
def forward_collision_warning(frame):
    objects = detect_objects(frame)
    for (box, confidence) in objects:
        x, y, w, h = box
        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Car: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Warning if object is too close (collision risk)
        if w > 150:  # Adjust threshold based on testing and calibration
            cv2.putText(frame, "Collision Warning!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    return frame

# Testing the FCW on video feed
cap = cv2.VideoCapture("test_video.mp4")  # Or replace with a camera index for live video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = forward_collision_warning(frame)
    cv2.imshow("Forward Collision Warning", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
