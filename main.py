import cv2 as cv
import time
import numpy as np
from driver import check_object, check_object_2
from kalman import KalmanFilter
import matplotlib.pyplot as plt
COLORS = [(None, None, None), (255, 0, 0), (255, 165, 0), (0, 255, 255)]

OBJECT = [(None, None, None), (0, 0, 255), (0, 255, 0), (0, 0, 128)]

# input class names - MS COCO

class_name = []
with open('yolo/classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]


starting_time = time.time()
# Initialize the model, backend and target
net = cv.dnn.readNet('yolo/yolov4-tiny.weights', 'yolo/yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)

# ball.mp4
model.setInputParams(size=(960, 540), scale=1/255, swapRB=True)
cap = cv.VideoCapture('input/ball.mp4')


def get_x():
    x = list()
    meas = list()
    frame_count = 0
    time_now = time.time()
    while True:
        ret, frame = cap.read()

        if ret == False:
            break

        _, _, boxes = model.detect(frame, 0.4, 0.4)

        for index, box in enumerate(boxes):

            # if len(x) == 2:
            #     meas.append((x[1] - x[0])/2)
            #     x.pop(0)
            # print(x.pop(0))

            centeroid = np.array([box[0] + (box[2] // 2),
                                  box[1] + (box[3]) // 2])

            x.append(centeroid[0])
            cv.rectangle(frame, box, OBJECT[1], 4)
            cv.putText(frame, f'Object {1}', (box[0], box[1]-10),
                       cv.FONT_HERSHEY_COMPLEX, 1, OBJECT[1], 1)

        # print("########### No Objects Detected ###########")
        # Kalman
        cv.imshow('frame', frame)
        frame_count += 1
        # video.write(frame)

        key = cv.waitKey(1)

        if key == ord('q'):
            break

    print(frame_count)
    print(len(x))
    # print(np.mean(meas))
    # print(frame_count / (end_time - time_now))

    return x


def main():
    dt = 8.9
    # Define a model track
    t = np.arange(0, 52, dt)
    print(t)
    real_track = get_x()
    u = 2
    # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    std_acc = 0.25
    std_meas = 1.2    # and standard deviation of the measurement is 1.2 (m)
    # create KalmanFilter object
    kf = KalmanFilter(dt, u, std_acc, std_meas)
    predictions = []
    measurements = []
    for x in real_track:
        # Mesurement
        z = kf.H * x + np.random.normal(0, 50)
        measurements.append(z.item(0))
        kf.predict(dt)
        predictions.append(kf.get_x)
        kf.update(z.item(0))
    fig = plt.figure()
    fig.suptitle(
        'Example of Kalman filter for tracking a moving object in 1-D', fontsize=20)
    plt.plot(t, measurements, label='Measurements', color='b', linewidth=0.5)
    plt.plot(t, np.array(real_track), label='Real Track',
             color='y', linewidth=1.5)
    plt.plot(t, np.squeeze(predictions),
             label='Kalman Filter Prediction', color='r', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
