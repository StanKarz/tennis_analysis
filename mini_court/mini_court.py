from utils import pixel_to_distance, distance_to_pixel
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
        return distance_to_pixel(meters, constants.DOUBLINE_WIDTH, self.court_drawing_width)

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

        # # get net midpoint (ensure the division is inside the int() conversion)
        # net_mid_x = int((net_start_point[0] + net_end_point[0]) / 2)
        # net_mid_y = int(net_start_point[1])  # Ensure it's an integer

        # # get service line position and cast it to int
        # service_line_y = int(self.court_start_y + self.convert_meteres_to_pixels(constants.SERVICE_LINE_LENGTH))

        # # Draw the line from the net's midpoint to the service line
        # cv2.line(frame, (net_mid_x, net_mid_y), (net_mid_x, service_line_y), (255, 0, 0), 2)

        return frame

    def draw_minicourt(self, frames):
        output_frames = [self.draw_court(self.draw_bg_rectangle(frame)) for frame in frames]
        return output_frames
