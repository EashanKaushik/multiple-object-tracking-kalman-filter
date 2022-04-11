import numpy as np
import cv2 as cv
import time
# IOU_THRESHOLD = 0.3  # for 1 ball
IOU_THRESHOLD = 0.45


def add_border(box, row, col):
    box_row = 0
    box_col = 0
    if row > box.shape[0]:
        box_row = row - box.shape[0]

    if col > box.shape[1]:
        box_col = col - box.shape[1]

    return cv.copyMakeBorder(
        box, box_row, 0, box_col, 0, cv.BORDER_CONSTANT, None, value=0)


def intersection_over_union(box1, box2):
    # print(
    #     f'box1 shape: {box1.shape} --- box2 shape: {box2.shape} --- equal? {box1.shape == box2.shape}')

    if box1.shape != box2.shape:
        row = max(box1.shape[0], box2.shape[0])
        col = max(box1.shape[1], box2.shape[1])
        channels = 3

        # for box 1
        box1 = add_border(box1, row, col)

        # for box 2
        box2 = add_border(box2, row, col)

    # print(
    #     f'AFTER: box1 shape: {box1.shape} --- box2 shape: {box2.shape} --- equal? {box1.shape == box2.shape}')

    if box1.shape == box2.shape:
        intersection = np.logical_and(box1, box2)
        union = np.logical_or(box1, box2)
        return np.sum(intersection) / np.sum(union)
    else:
        raise Exception


def check_object(frame, box, objects, next_object_id, total_boxes_detected):
    if len(objects) == 0:
        # first object
        print('No Objects')
        new_crop_img = frame[box[1]: box[1] + box[3], box[0]: box[0]+box[2]]
        objects[next_object_id] = new_crop_img

        return next_object_id + 1, objects, next_object_id
    else:
        # not the first object
        new_crop_img = frame[box[1]: box[1] + box[3], box[0]: box[0]+box[2]]

        if len(objects) >= total_boxes_detected:
            is_assigned = False
            assigned = dict()

            for object_id, stored_crop_img in objects.items():
                iou_score = intersection_over_union(
                    stored_crop_img, new_crop_img)
                if iou_score >= IOU_THRESHOLD:
                    assigned[iou_score] = object_id
                    is_assigned = True

            if is_assigned:
                # object was assigned more than one ids
                largest_iou = sorted(assigned.keys())[-1]

                objects[assigned[largest_iou]] = new_crop_img

                return next_object_id, objects, assigned[largest_iou]

            else:
                # object was not assigned any id
                # create new object
                raise Exception()

                objects[next_object_id] = new_crop_img

                return next_object_id + 1, objects, next_object_id

        else:
            time.sleep(5)

            objects[next_object_id] = new_crop_img

            return next_object_id + 1, objects, next_object_id
