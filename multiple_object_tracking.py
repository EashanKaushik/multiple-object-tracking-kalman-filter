import cv2 as cv
import time
import numpy as np
from driver import check_object
from kalman_filter import KalmanFilter

# COLORS
COLORS = [(None, None, None), (255, 0, 0), (255, 165, 0), (0, 255, 255)]
OBJECT = [(None, None, None), (0, 0, 255), (0, 255, 0), (0, 0, 128)]

# YOLO weights and config file
YOLO_WEIGHTS = 'yolo/yolov4-tiny.weights'
YOLO_CONF = 'yolo/yolov4-tiny.cfg'


def set_yolo_model() -> cv:
    """For setting YOLO model

    Returns:
        cv: Configured YOLO Model
    """

    # Initialize the model, backend and target
    net = cv.dnn.readNet(YOLO_WEIGHTS, YOLO_CONF)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(net)

    return model


def mot(object_conf: dict) -> None:
    """For Object Detection using YOLO and Object Tracking using Kalman Filter

    Args:
        object_conf (dict): Video object, Hyperparameters
    """

    # get the YOLO Model
    model = set_yolo_model()

    # set_model_params
    model.setInputParams(
        size=object_conf['input_size'], scale=1/255, swapRB=True)

    # input the video
    cap = cv.VideoCapture(object_conf['input'])

    # save the video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(
        object_conf['output'], fourcc, 10, object_conf['output_size'])

    # start time
    starting_time = time.time()

    # storing objects
    objects = dict()
    # next available object
    next_object_id = 1
    # kalman filter dict
    kf = dict()
    # current object assigned
    current_object_id = 0

    while True:
        # read the frame
        ret, frame = cap.read()

        # calculate dt
        new_time = time.time()
        dt = (new_time - starting_time)
        starting_time = new_time

        if ret == False:
            break

        # get bounding box from YOLO
        _, _, boxes = model.detect(
            frame, object_conf['Conf_threshold'], object_conf['NMS_threshold'])

        object_id_detected = list()

        print('\n\nStart of new Frame')

        # iterate over all the boxes detected
        for index, box in enumerate(boxes):

            # centroid of the current box
            centeroid = np.array([box[0] + (box[2] // 2),
                                  box[1] + (box[3]) // 2])

            # check if centroid belong to existing object id else create new object id
            next_object_id, objects, current_object_id = check_object(
                centeroid, objects, next_object_id)

            # detected object id
            object_id_detected.append(current_object_id)

            # create a circle arounf the centroid
            cv.circle(frame, centeroid, 3, OBJECT[current_object_id], 3)

            # check if current_object_id belongs to existing object ids
            if current_object_id not in kf.keys():
                # current_object_id does not belong to existing object ids

                # create new KalmanFilter instance
                kf[current_object_id] = KalmanFilter(
                    x=centeroid[0], u=10, std_acc=0.1, std_meas=0.1)

                # draw circle arounf kalman filter prediction
                cv.circle(frame, (kf[current_object_id].get_x,
                                  centeroid[1]), 1, COLORS[current_object_id], 3)

            else:
                # current_object_id belongs to existing object ids

                # predict current x
                kf[current_object_id].predict(dt)

                # draw circle arounf kalman filter prediction
                cv.circle(frame, (kf[current_object_id].get_x,
                                  centeroid[1]), 1, COLORS[current_object_id], 3)

                # update kalman filter with ground truth
                kf[current_object_id].update(centeroid[0])

            print(
                f'Object {current_object_id} x when detected & updated: {kf[current_object_id].get_x}')

            # create rectangle around detected object
            cv.rectangle(frame, box, OBJECT[current_object_id], 4)
            # Put object id around object
            cv.putText(frame, f'Object {current_object_id}', (box[0], box[1]-10),
                       cv.FONT_HERSHEY_COMPLEX, 1, OBJECT[current_object_id], 1)

        # check if any object was not detected in current frame
        for object_id, object_kf in kf.items():

            if object_id not in object_id_detected:
                # if any object was not detected in current frame

                # predict current x
                object_kf.predict(dt)

                # update object centroid in objects dict
                objects[object_id] = np.array([object_kf.get_x, centeroid[1]])

                # # draw circle arounf kalman filter prediction
                cv.circle(
                    frame, (object_kf.get_x, centeroid[1]), 1, COLORS[object_id], 3)

                print(
                    f'Object {object_id} x when not detected: {object_kf.get_x}')

        print('End of Frame\n\n')

        # show frame
        cv.imshow('frame', frame)

        # write frame
        video.write(frame)

        key = cv.waitKey(4)

        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    video.release()
