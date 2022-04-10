import cv2 as cv
import time
import numpy as np
from kalman_filter import KalmanFilter

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
# model.setInputParams(size=(600, 500), scale=1/255, swapRB=True)
# cap = cv.VideoCapture('input/objectTracking_examples_multiObject.avi')

# KF = list()

starting_time = time.time()

initialize = True

kf = None

while True:
    ret, frame = cap.read()

    # not the first frame
    new_time = time.time()
    dt = (new_time - starting_time)
    starting_time = new_time

    if ret == False:
        break

    _, _, boxes = model.detect(frame, 0.4, 0.4)

    if len(boxes) != 0:
        for box in boxes:
            centeroid = np.array([box[0] + (box[2] // 2),
                                  box[1] + (box[3]) // 2])
            cv.circle(frame, centeroid, 3, (0, 255, 0), 3)

            if initialize:
                initialize = False
                kf = KalmanFilter(centeroid[0], centeroid[1])
                kf.predict(dt)

                print(f'Kalman Filter Predictions: x:{kf.x}, y:{kf.y}')
                print(f'Ground Truth: x:{centeroid[0]}, y:{centeroid[1]}')

                cv.circle(frame, (kf.x, kf.y), 1, (255, 0, 0), 3)

            else:
                kf.predict(dt)

                print(f'Kalman Filter Predictions: x:{kf.x}, y:{kf.y}')
                print(f'Ground Truth: x:{centeroid[0]}, y:{centeroid[1]}')

                cv.circle(frame, (kf.x, kf.y), 1, (255, 0, 0), 3)

                kf.update(centeroid)

            cv.rectangle(frame, box, (0, 0, 255), 4)
    elif kf:

        kf.predict(dt)

        cv.circle(frame, (kf.x, kf.y), 1, (255, 0, 0), 3)

        kf.update(centeroid)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

cap.release()

cv.destroyAllWindows()
