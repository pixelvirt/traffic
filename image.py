import cv2
import numpy as np
from collections import Counter
import webcolors
from sklearn.cluster import KMeans

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def dominant_color(frame, k=3):
    pixels = frame.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    palette = np.uint8(centers)
    return palette[np.argmax(np.unique(labels, return_counts=True)[1])]

def get_dominant_color(frame, k=3):
    color = dominant_color(frame, k)
    return color

def detect_and_color_recognition(image, yolo):
    (H, W) = image.shape[:2]
    ln = yolo.getLayerNames()
    ln = [ln[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo.setInput(blob)
    layer_outputs = yolo.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = get_dominant_color(image[y:y + h, x:x + w], k=3)
            draw_prediction(image, classIDs[i], confidences[i], x, y, x + w, y + h)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image

def main():
    # Load the COCO class labels
    classes = open("yolo-coco/coco.names").read().strip().split("\n")

    # Load the YOLO model
    yolo = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg")

    # Load the image
    image = cv2.imread("shirt.jpeg")

    # Perform object detection and color recognition on the image
    color = detect_and_color_recognition(image, yolo)
    print(color)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()