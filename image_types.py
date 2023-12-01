import cv2
import numpy as np
import time

# Load YOLO network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the blob and run forward pass through YOLO
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def detect_objects(image, net, outputLayers):
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputLayers)
    return outs

def get_bounding_box(outs, height, width, confidence_threshold):
    class_ids = []
    confidencess = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidencess.append(float(confidence))
                boxes.append([x, y, w, h])
    return boxes, class_ids, confidencess

def draw_labels_and_boxes(image, boxes, class_ids, confidencess, class_names):
    indexes = cv2.dnn.NMSBoxes(boxes, confidencess, confidence_threshold, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidencess[i]:.2f}"
            color = (255, 255, 255)
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(image, label, (int(x), int(y) - 5), font, 1, color, 2)
    return image

# Load the input image
image = cv2.imread("input.jpg")
height, width, channels = image.