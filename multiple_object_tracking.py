import cv2 as cv
import time
import numpy as np
from driver import check_object
from kalman_filter import KalmanFilter

COLORS = [(None, None, None), (255, 0, 0), (255, 165, 0), (0, 255, 255)]

OBJECT = [(None, None, None), (0, 0, 255), (0, 255, 0), (0, 0, 128)]

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
# model.setInputParams(size=(960, 540), scale=1/255, swapRB=True)
# cap = cv.VideoCapture('input/ball.mp4')
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# video = cv.VideoWriter('video.avi', fourcc, 10, (960, 540))

# objectTracking_examples_multiObject.avi
model.setInputParams(size=(600, 500), scale=1/255, swapRB=True)
cap = cv.VideoCapture('input/objectTracking_examples_multiObject.avi')
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# video = cv.VideoWriter('video2.avi', fourcc, 10, (960, 540))


starting_time = time.time()

objects = dict()
next_object_id = 1
kf = dict()
current_object_id = 0

while True:
    ret, frame = cap.read()

    new_time = time.time()
    dt = (new_time - starting_time)
    starting_time = new_time

    if ret == False:
        break

    _, _, boxes = model.detect(frame, 0.4, 0.4)

    if len(boxes) > 2:
        continue

    if len(boxes) != 0:

        print(
            f"########### Objects Detected {len(boxes)}, {boxes} ###########")
        for index, box in enumerate(boxes):

            next_object_id, objects, current_object_id = check_object(
                frame, box, objects, next_object_id, len(boxes))

            print(f'Current Object: {current_object_id}')
            # print(objects.keys())

            centeroid = np.array([box[0] + (box[2] // 2),
                                  box[1] + (box[3]) // 2])

            cv.circle(frame, centeroid, 3, OBJECT[current_object_id], 3)

            if current_object_id not in kf.keys():
                # Kalman
                kf[current_object_id] = KalmanFilter(
                    initial_pos=centeroid[0], initial_velocity=0, acceleration=2)
                # print(
                #     f'Kalman Filter Predictions: x:{kf[current_object_id].x_float}, y:{centeroid[1]}')
                # print(f'Ground Truth: x:{centeroid[0]}, y:{centeroid[1]}')

                cv.circle(frame, (kf[current_object_id].x,
                          centeroid[1]), 1, COLORS[current_object_id], 3)

            else:
                # Kalman
                kf[current_object_id].predict(dt)
                # print(
                #     f'Kalman Filter Predictions: x:{kf[current_object_id].x_float}, y:{centeroid[1]}')
                # print(f'Ground Truth: x:{centeroid[0]}, y:{centeroid[1]}')

                cv.circle(frame, (kf[current_object_id].x,
                          centeroid[1]), 1, COLORS[current_object_id], 3)

                kf[current_object_id].update(centeroid[0],  0.37)

            cv.rectangle(frame, box, OBJECT[current_object_id], 4)
            cv.putText(frame, f'Object {current_object_id}', (box[0], box[1]-10),
                       cv.FONT_HERSHEY_COMPLEX, 1, OBJECT[current_object_id], 1)
    elif len(kf) != 0:
        print("########### No Objects Detected ###########")
        # Kalman
        for object_id, object_kf in kf.items():

            object_kf.predict(dt)
            # print(
            #     f'Kalman Filter Predictions: x:{object_kf.x_float}, y:{centeroid[1]}')
            # print(f'Ground Truth: x:{centeroid[0]}, y:{centeroid[1]}')

            cv.circle(
                frame, (object_kf.x, centeroid[1]), 1, COLORS[object_id], 3)

    cv.imshow('frame', frame)
    print("########### Objects over ###########")
    # video.write(frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

cap.release()

cv.destroyAllWindows()
# video.release()
