import sys
import numpy as np

DISTANCE_THRESHOLD = 200


def check_object(centroid: int, objects: dict, next_object_id: int) -> (int, dict, int):
    """Assings object id based on distance between current and previous centroid

    Args:
        centroid (int): centroid of current object
        objects (dict): dictionary of object id and previously known objects
        next_object_id (int): next available object id

    Returns:
        (int, dict, int): (next available object id, updated objects dictionary, current object id assigned)
    """

    if len(objects) == 0:
        # first object
        objects[next_object_id] = centroid
        return next_object_id + 1, objects, next_object_id
    else:
        # not the first object

        minimum_distance = sys.maxsize
        is_assigned = False

        # find minimum distance
        for object_id, object_centroid in objects.items():
            dist = np.linalg.norm(centroid - object_centroid)
            if dist < minimum_distance:
                minimum_distance = dist
                assigned_object_id = object_id
                assigned_distance = dist
                is_assigned = True

        # check if minimum is less than the threshold
        if is_assigned and assigned_distance < DISTANCE_THRESHOLD:
            #  minimum is less than threshold
            objects[assigned_object_id] = centroid

            return next_object_id, objects, assigned_object_id
        else:
            # minimum is not less than the threshold
            objects[next_object_id] = centroid
            return next_object_id + 1, objects, next_object_id
