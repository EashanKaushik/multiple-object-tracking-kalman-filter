import cv2 as cv
import time

# input class names - MS COCO
class_name = []
with open('yolo/classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]


# Initialize the model, backend and target
net = cv.dnn.readNet('yolo/yolov4-tiny.weights', 'yolo/yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)

# ball.mp4
model.setInputParams(size=(960, 540), scale=1/255, swapRB=True)
cap = cv.VideoCapture('input/ball.mp4')

# objectTracking_examples_multiObject.avi
# model.setInputParams(size=(600, 500), scale=1/255, swapRB=False)
# cap = cv.VideoCapture('input/objectTracking_examples_multiObject.avi')

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    _, _, boxes = model.detect(frame, 0.4, 0.4)

    for box in boxes:

        cv.rectangle(frame, box, (0, 0, 255), 1)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

cap.release()

cv.destroyAllWindows()
