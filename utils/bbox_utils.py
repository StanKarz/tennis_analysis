def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2)) / 2
    y_center = int((y1 + y2)) / 2
    return (x_center, y_center)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_foot_pos(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), y2


def get_closest_kp_idx(point, kps, kp_indices):
    min_distance = float("inf")
    closest_kp_idx = kp_indices[0]
    for kp_idx in kp_indices:
        kp = (kps[kp_idx * 2], kps[kp_idx * 2 + 1])
        distance = abs(point[1] - kp[1])

        if distance < min_distance:
            min_distance = distance
            closest_kp_idx = kp_idx

    return closest_kp_idx


def get_bbox_height(bbox):
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def get_center_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
