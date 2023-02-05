from shapely.geometry import Polygon
import numpy as np
import time

def bbox_to_centers(bboxes):
    """
    convert (top, right, bottom, left) to (y_center, x_center)
    :return: ndarray with shape of n*2
    """
    return np.vstack(((bboxes[:, 0] + bboxes[:, 2])/2, (bboxes[:, 1] + bboxes[:, 3])/2)).T

def convert_yolo_detections(detections):
    """
    convert yolo detection format to person top, right, bottom, left bbox
    :param detections:
    """
    loactions = []
    for det in detections:
        if det["name"] == "person":
            loactions.append((det["box_points"][1], det["box_points"][2], det["box_points"][3], det["box_points"][0]))
    return loactions

def  is_only_one_match(in_mat, axis=0):
    """
    check if there is maximum True values in each row or column
    :param iou_mat: input matrix to check
    :param axis: axis to check on
    :return: True if there is a maximum of one True in the specified rows/columns. if the answer is now it returm vecotor which tell where
    is the duplicats trues (oredr vector).  for new face with no overlapping with previous face return -1 in the order vector
    """
    number_of_ones = np.sum(in_mat, axis=axis)
    in_place_participants = np.sum(number_of_ones == 1)
    is_only_one_one = np.all(number_of_ones <= 1)
    order = np.argmax(in_mat, axis=axis)
    order = np.where(number_of_ones == 0, -1, order)
    order = np.where(number_of_ones > 1, -2, order)
    return is_only_one_one, in_place_participants, order

def face_location_to_corners(face_location):
    (top, right, bottom, left) = face_location
    return [[top, left], [bottom, left], [bottom, right], [top, right]]


def iou_one_box(box_1, box_2, convention="trbl"):
    if convention == "trbl":
        xmin = 3;        ymin = 0;      xmax = 1;       ymax = 2
    elif convention == "tlbr":
        xmin = 1;       ymin = 0;       xmax = 3;       ymax = 2
    elif convention == "ltrb":
        xmin = 0;        ymin = 1;      xmax = 2;       ymax = 3
    min_xy = np.maximum(box_1[[xmin, ymin]], box_2[[xmin, ymin]])
    max_xy = np.minimum(box_1[[xmax, ymax]], box_2[[xmax, ymax]])

    # Compute the side lengths of the intersection rectangles.
    intersection_area = np.prod(np.maximum(0, max_xy - min_xy))

    boxes1_areas = (box_1[xmax] - box_1[xmin]) * (box_1[ymax] - box_1[ymin])
    boxes2_areas = (box_2[xmax] - box_2[xmin]) * (box_2[ymax] - box_2[ymin])

    union_areas = boxes1_areas + boxes2_areas - intersection_area

    return intersection_area / union_areas

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    return poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

def intersection_area(boxes1, boxes2, convention="trbl"):
    """
    :param boxes1: 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes. coordinates format: top, right, bottom, left
    :param boxes2: 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes. coordinates format: top, right, bottom, left
    :param convention: order of box corners. "ltrb" = left, top, right, bottom, "tlbr" = top, left, bottom, right
    :return: `(m,n)` matrix with the intersection areas for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`.
    """
    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)
    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`
    if convention == "trbl":
        xmin = 3;        ymin = 0;        xmax = 1;        ymax = 2
    elif convention == "tlbr":
        xmin = 1;       ymin = 0;         xmax = 3;        ymax = 2
    elif convention == "ltrb":
        xmin = 0;       ymin = 1;         xmax = 2;        ymax = 3

    # For all possible box combinations, get the greater xmin and ymin values.
    # This is a tensor of shape (m,n,2).
    min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                        np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))

    # For all possible box combinations, get the smaller xmax and ymax values.
    # This is a tensor of shape (m,n,2).
    max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                        np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

    # Compute the side lengths of the intersection rectangles.
    side_lengths = np.maximum(0, max_xy - min_xy)

    return side_lengths[:, :, 0] * side_lengths[:, :, 1]


def iou(boxes1, boxes2, convention="trbl"):
    """
    :param boxes1: 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes. coordinates format: top, right, bottom, left
    :param boxes2: 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes. coordinates format: top, right, bottom, left
    :param convention: order of box corners. "ltrb" = left, top, right, bottom, "tlbr" = top, left, bottom, right
    :return: `(m,n)` matrix with the iou for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`.
    """
    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    intersection_areas = intersection_area(boxes1, boxes2)

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`
    if convention == "trbl":
        xmin = 3;   ymin = 0;     xmax = 1;    ymax = 2
    elif convention == "tlbr":
        xmin = 1;    ymin = 0;    xmax = 3;    ymax = 2
    elif convention == "ltrb":
        xmin = 0;    ymin = 1;    xmax = 2;    ymax = 3

    boxes1_areas = np.tile(
        np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin]) * (boxes1[:, ymax] - boxes1[:, ymin]), axis=1),
        reps=(1, n))
    boxes2_areas = np.tile(
        np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin]) * (boxes2[:, ymax] - boxes2[:, ymin]), axis=0),
        reps=(m, 1))

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


def inflate_locations(boxes, frame_shape, inflate_ratio = 1.1, convention = "trbl"):
    if convention == "trbl":
        xmin = 3;   ymin = 0;     xmax = 1;    ymax = 2
    elif convention == "tlbr":
        xmin = 1;    ymin = 0;    xmax = 3;    ymax = 2
    elif convention == "ltrb":
        xmin = 0;    ymin = 1;    xmax = 2;    ymax = 3

    w = (boxes[:, xmax] - boxes[:, xmin]) * (inflate_ratio - 1)
    h = (boxes[:, ymax] - boxes[:, ymin]) * (inflate_ratio - 1)
    inflated = np.zeros_like(boxes)
    inflated[:, xmax] = np.where(boxes[:, xmax] + np.ceil(w/2) > frame_shape[1] , frame_shape[1] , boxes[:, xmax] + np.ceil(w/2))
    inflated[:, xmin] = np.where(boxes[:, xmin] - np.ceil(w/2) < 0 , 0 , boxes[:, xmin] - np.ceil(w/2))
    inflated[:, ymax] = np.where(boxes[:, ymax] + np.ceil(h/2) > frame_shape[0] , frame_shape[0] , boxes[:, ymax] + np.ceil(h/2))
    inflated[:, ymin] = np.where(boxes[:, ymin] - np.ceil(h/2) < 0 , 0 , boxes[:, ymin] - np.ceil(h/2))
    return inflated


def test_iou():
    flag_random_boxes = True
    if flag_random_boxes:
        m=25
        n=25
        corners1 = np.round(10*np.random.rand(m,2))
        corners2 = np.round(10*np.random.rand(n, 2))
        boxes1 = np.append(corners1, corners1 + np.ceil(10 * np.random.rand(m ,2)), axis=1)
        boxes2 = np.append(corners2, corners2 + np.ceil(10 * np.random.rand(n, 2)), axis=1)
    else:
        boxes1 = np.array([[0,0,10,10], [5, 5, 15, 15.1], [20,20,30,30]])
        boxes2 = np.array([[5, 5, 15, 15], [14, 14, 30, 30]])
        m = boxes1.shape[0]  # The number of boxes in `boxes1`
        n = boxes2.shape[0]

    iou_one_box(boxes1[0, :], boxes2[0, :])

    start_analyze_time1 = time.time()
    res_iou1 = iou(boxes1, boxes2)
    end_analyze_time1 = time.time()

    res_iou2 = np.zeros((m, n))
    start_analyze_time2 = time.time()
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            res_iou2[i,j] = iou_one_box(box1, box2)
    end_analyze_time2 = time.time()
    print(np.all(res_iou1 == res_iou2))
    print(f"time for vectorized ioc calculation: {end_analyze_time1-start_analyze_time1:.4f} sec")
    print(f"time for double-loop ioc calculation: {end_analyze_time2 - start_analyze_time2:.4f} sec")
    print(f"improvement ratio : {(end_analyze_time2 - start_analyze_time2) / (end_analyze_time1-start_analyze_time1)}")


def get_face_index(faces_dict, face_location, iou_thresh=0.3, location_name="location"):
    if not faces_dict:
        return -1
    for k, face_features in faces_dict.items():
        current_iou = calculate_iou(face_location_to_corners(face_location), face_location_to_corners(face_features[location_name]))
        if current_iou > iou_thresh:
            return k
    return -1


def get_iou_match(faces_locations, current_face_location):
    # TODO: vectorize this function
    iou = []
    for stored_location in faces_locations:
        current_iou = calculate_iou(face_location_to_corners(current_face_location), face_location_to_corners(stored_location))
        iou.append(current_iou)
    return iou


