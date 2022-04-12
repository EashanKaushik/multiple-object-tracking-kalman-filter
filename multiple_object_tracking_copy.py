import cv2 as cv
import time
import numpy as np
from driver import check_object, check_object_2
from kalman import KalmanFilter

COLORS = [(None, None, None), (255, 0, 0), (255, 165, 0), (0, 255, 255)]

OBJECT = [(None, None, None), (0, 0, 255), (0, 255, 0), (0, 0, 128)]

# input class names - MS COCO
class_name = []
with open('yolo/classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]


# Initialize the model, backend and target
net = cv.dnn.readNet('yolo/yolov4-tiny.weights', 'yolo/yolov4-tiny.cfg')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)

# ball.mp4
# model.setInputParams(size=(960, 540), scale=1/255, swapRB=True)
# cap = cv.VideoCapture('input/ball.mp4')
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# video = cv.VideoWriter('video.avi', fourcc, 10, (960, 540))

# objectTracking_examples_multiObject.avi
model.setInputParams(size=(500, 500), scale=1/255, swapRB=True)
cap = cv.VideoCapture('input/objectTracking_examples_multiObject.avi')
fourcc = cv.VideoWriter_fourcc(*'mp4v')
video = cv.VideoWriter('video2.avi', fourcc, 10, (640, 360))


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

    _, _, boxes = model.detect(frame, 0.3, 0.4)
    print('\n\nStart of new Frame')
    object_id_detected = list()
    for index, box in enumerate(boxes):

        # next_object_id, objects, current_object_id = check_object(
        #     frame, box, objects, next_object_id, len(boxes))

        centeroid = np.array([box[0] + (box[2] // 2),
                              box[1] + (box[3]) // 2])
        print(centeroid[0])

        next_object_id, objects, current_object_id = check_object_2(
            centeroid, objects, next_object_id)

        object_id_detected.append(current_object_id)

        cv.circle(frame, centeroid, 3, OBJECT[current_object_id], 3)

        if current_object_id not in kf.keys():
            # Kalman
            kf[current_object_id] = KalmanFilter(
                # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
                x=centeroid[0], u=10, std_acc=0.1, std_meas=0.1)
            # print(
            #     f'Kalman Filter Predictions: x:{kf[current_object_id].x_float}, y:{centeroid[1]}')
            # print(f'Ground Truth: x:{centeroid[0]}, y:{centeroid[1]}')

            cv.circle(frame, (kf[current_object_id].get_x,
                              centeroid[1]), 1, COLORS[current_object_id], 3)

        else:
            # Kalman
            kf[current_object_id].predict(dt)
            # print(
            #     f'Kalman Filter Predictions: x:{kf[current_object_id].x_float}, y:{centeroid[1]}')
            # print(f'Ground Truth: x:{centeroid[0]}, y:{centeroid[1]}')

            cv.circle(frame, (kf[current_object_id].get_x,
                              centeroid[1]), 1, COLORS[current_object_id], 3)

            kf[current_object_id].update(centeroid[0])

        print(
            f'Object {current_object_id} x when detected & updated: {kf[current_object_id].get_x}')

        cv.rectangle(frame, box, OBJECT[current_object_id], 4)
        cv.putText(frame, f'Object {current_object_id}', (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 1, OBJECT[current_object_id], 1)

    # print("########### No Objects Detected ###########")
    # Kalman
    for object_id, object_kf in kf.items():

        if object_id not in object_id_detected:

            object_kf.predict(dt)

            objects[object_id] = np.array([object_kf.get_x, centeroid[1]])
            cv.circle(
                frame, (object_kf.get_x, centeroid[1]), 1, COLORS[object_id], 3)

            print(f'Object {object_id} x when not detected: {object_kf.get_x}')

    print('End of Frame\n\n')
    cv.imshow('frame', frame)
    video.write(frame)

    key = cv.waitKey(4)

    if key == ord('q'):
        break

print(objects)

cap.release()

cv.destroyAllWindows()
video.release()
