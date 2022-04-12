
from multiple_object_tracking import mot

MULTI_OBJECT = {
    "input": "input/objectTracking_examples_multiObject.avi",
    "input_size": (500, 500),
    "output": "output/multiple_object_with_occlusion.avi",
    "output_size": (640, 360),
    "Conf_threshold": 0.3,
    "NMS_threshold": 0.4,
}

SINGLE_OBJECT = {
    "input": "input/ball.mp4",
    "input_size": (960, 540),
    "output": "output/single_object_with_occlusion.avi",
    "output_size": (960, 540),
    "Conf_threshold": 0.4,
    "NMS_threshold": 0.4,
}


if __name__ == '__main__':
    mot(MULTI_OBJECT)

    mot(SINGLE_OBJECT)
