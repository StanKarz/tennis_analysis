from utils import (
    pixel_to_distance,
    distance_to_pixel,
    get_foot_pos,
    get_closest_kp_idx,
    get_bbox_height,
    measure_xy_distance,
    get_center_bbox,
    measure_distance,
)
import constants
import cv2
import sys
import numpy as np

sys.path.append("../")


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectange_width = 250
        self.drawing_rectange_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvs_bg_box_pos(frame)
        self.set_minicourt_pos()
        self.set_court_drawing_kps()
        self.set_court_lines()

    def convert_meteres_to_pixels(self, meters):
        return distance_to_pixel(meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)

    def set_court_drawing_kps(self):
        drawing_kps = [0] * 28

        # point 0
        drawing_kps[0], drawing_kps[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_kps[2], drawing_kps[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_kps[4] = int(self.court_start_x)
        drawing_kps[5] = self.court_start_y + self.convert_meteres_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_kps[6] = drawing_kps[0] + self.court_drawing_width
        drawing_kps[7] = drawing_kps[5]
        # point 4
        drawing_kps[8] = drawing_kps[0] + self.convert_meteres_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_kps[9] = drawing_kps[1]
        # point 5
        drawing_kps[10] = drawing_kps[4] + self.convert_meteres_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_kps[11] = drawing_kps[5]
        # point 6
        drawing_kps[12] = drawing_kps[2] - self.convert_meteres_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_kps[13] = drawing_kps[3]
        # point 7
        drawing_kps[14] = drawing_kps[6] - self.convert_meteres_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_kps[15] = drawing_kps[7]
        # point 8
        drawing_kps[16] = drawing_kps[8]
        drawing_kps[17] = drawing_kps[9] + self.convert_meteres_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 9
        drawing_kps[18] = drawing_kps[16] + self.convert_meteres_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_kps[19] = drawing_kps[17]
        # point 10
        drawing_kps[20] = drawing_kps[10]
        drawing_kps[21] = drawing_kps[11] - self.convert_meteres_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_kps[22] = drawing_kps[20] + self.convert_meteres_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_kps[23] = drawing_kps[21]
        # point 12
        drawing_kps[24] = int((drawing_kps[16] + drawing_kps[18]) / 2)
        drawing_kps[25] = drawing_kps[17]
        # point 13
        drawing_kps[26] = int((drawing_kps[20] + drawing_kps[22]) / 2)
        drawing_kps[27] = drawing_kps[21]

        self.drawing_kps = drawing_kps

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3),
        ]

    def set_minicourt_pos(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvs_bg_box_pos(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectange_height
        self.start_x = self.end_x - self.drawing_rectange_width
        self.start_y = self.end_y - self.drawing_rectange_height

    def draw_bg_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            cv2.FILLED,
        )
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_kps), 2):
            x = int(self.drawing_kps[i])
            y = int(self.drawing_kps[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_kps[line[0] * 2]), int(self.drawing_kps[line[0] * 2 + 1]))
            end_point = (int(self.drawing_kps[line[1] * 2]), int(self.drawing_kps[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # draw net
        net_start_point = (
            self.drawing_kps[0],
            int((self.drawing_kps[1] + self.drawing_kps[5]) / 2),
        )
        net_end_point = (self.drawing_kps[2], int((self.drawing_kps[1] + self.drawing_kps[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        # get net midpoint (ensure the division is inside the int() conversion)
        net_mid_x = int((net_start_point[0] + net_end_point[0]) / 2)
        net_mid_y = int(net_start_point[1])  # Ensure it's an integer

        # get the position of the red point (which is drawing_kps[24], drawing_kps[25])

        # get service line position and cast it to int
        centerline_x1 = int(self.drawing_kps[24])
        centerline_y1 = int(self.drawing_kps[25])

        centerline_x2 = int(self.drawing_kps[26])
        centerline_y2 = int(self.drawing_kps[27])

        # Draw the line from the net's midpoint to the service line
        cv2.line(frame, (net_mid_x, net_mid_y), (centerline_x1, centerline_y1), (0, 0, 0), 2)
        cv2.line(frame, (net_mid_x, net_mid_y), (centerline_x2, centerline_y2), (0, 0, 0), 2)

        return frame

    def draw_minicourt(self, frames):
        output_frames = [self.draw_court(self.draw_bg_rectangle(frame)) for frame in frames]
        return output_frames

    def get_start_point_minicourt(self):
        return self.court_start_x, self.court_start_y

    def get_width_minicourt(self):
        return self.court_drawing_width

    def get_court_drawing_kps(self):
        return self.drawing_kps

    def get_minicourt_coords(self, object_pos, closest_kp, closest_kp_index, player_height_px, player_height_in_meters):

        # get the distance between the closest keypoint and the player
        distance_from_kp_x_px, distance_from_kp_y_px = measure_xy_distance(object_pos, closest_kp)

        # convert pixel distance to meters

        distance_from_kp_x_meters = pixel_to_distance(distance_from_kp_x_px, player_height_in_meters, player_height_px)

        distance_from_kp_y_meters = pixel_to_distance(distance_from_kp_y_px, player_height_in_meters, player_height_px)

        # convert to minicourt coordinates

        minicourt_x_distance_pixels = self.convert_meteres_to_pixels(distance_from_kp_x_meters)
        minicourt_y_distance_pixels = self.convert_meteres_to_pixels(distance_from_kp_y_meters)

        closest_minicourt_kp = self.drawing_kps[closest_kp_index * 2], self.drawing_kps[closest_kp_index * 2 + 1]

        minicourt_player_pos = (
            closest_minicourt_kp[0] + minicourt_x_distance_pixels,
            closest_minicourt_kp[1] + minicourt_y_distance_pixels,
        )

        return minicourt_player_pos

    def convert_bb_to_minicourt_coordinates(self, player_boxes, ball_boxes, original_court_kps):
        player_heights = {1: constants.PLAYER_1_HEIGHT, 2: constants.PLAYER_2_HEIGHT}

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_pos = get_center_bbox(ball_box)
            closest_player_id_to_ball = min(
                player_bbox.keys(), key=lambda x: measure_distance(ball_pos, get_center_bbox(player_bbox[x]))
            )

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_pos = get_foot_pos(bbox)

                # get closest kp in pixels
                closest_kp_idx = get_closest_kp_idx(foot_pos, original_court_kps, [0, 2, 12, 13])
                closest_kp = original_court_kps[closest_kp_idx * 2], original_court_kps[closest_kp_idx * 2 + 1]

                # get player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)

                bboxes_height_in_px = [
                    get_bbox_height(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)
                ]

                max_player_height_in_px = max(bboxes_height_in_px)

                minicourt_player_pos = self.get_minicourt_coords(
                    foot_pos, closest_kp, closest_kp_idx, max_player_height_in_px, player_heights[player_id]
                )

                output_player_bboxes_dict[player_id] = minicourt_player_pos

                if closest_player_id_to_ball == player_id:
                    # get closest kp in px
                    closest_kp_idx = get_closest_kp_idx(ball_pos, original_court_kps, [0, 2, 12, 13])
                    closest_kp = original_court_kps[closest_kp_idx * 2], original_court_kps[closest_kp_idx * 2 + 1]

                    minicourt_player_pos = self.get_minicourt_coords(
                        ball_pos, closest_kp, closest_kp_idx, max_player_height_in_px, player_heights[player_id]
                    )

                    output_ball_boxes.append({1: minicourt_player_pos})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points_on_minicourt(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)
        return frames
