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
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


sys.path.append("../")


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectange_width = 250
        self.drawing_rectange_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.heatmap_resolution = (100, 200)  # can be adjusted
        self.heatmaps = {1: np.zeros(self.heatmap_resolution), 2: np.zeros(self.heatmap_resolution)}

        self.set_canvs_bg_box_pos(frame)
        self.set_minicourt_pos()
        self.set_court_drawing_kps()
        self.set_court_lines()

    def update_heatmap(self, player_position):
        for player_id, position in player_position.items():
            x, y = position
            x_norm = (x - self.court_start_x) / self.court_drawing_width
            y_norm = (y - self.court_start_y) / (self.court_end_y - self.court_start_y)

            if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                x_idx = int(x_norm * (self.heatmap_resolution[1] - 1))
                y_idx = int(y_norm * (self.heatmap_resolution[0] - 1))
                self.heatmaps[player_id][y_idx, x_idx] += 1

        # Add debug print statements
        print(f"Player 1 heatmap sum: {np.sum(self.heatmaps[1])}")
        print(f"Player 2 heatmap sum: {np.sum(self.heatmaps[2])}")

    def generate_heatmap_overlay(self, player_id):
        heatmap = gaussian_filter(self.heatmaps[player_id], sigma=1)
        heatmap = (heatmap - heatmap.min()) / (
            heatmap.max() - heatmap.min() + 1e-8
        )  # Add small value to avoid division by zero

        # Increase the intensity of the heatmap
        heatmap = np.power(heatmap, 0.1)  # This will make lower values more visible

        colors = [(0, 0, 0, 0), (0, 0, 1, 0.8), (0, 1, 0, 0.9), (1, 0, 0, 1)]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

        plt.figure(figsize=(10, 5))
        plt.imshow(heatmap, cmap=cmap)
        plt.axis("off")

        temp_file = f"temp_heatmap_{player_id}.png"
        plt.savefig(temp_file, format="png", bbox_inches="tight", pad_inches=0, transparent=True, dpi=300)
        plt.close()

        heatmap_img = cv2.imread(temp_file, cv2.IMREAD_UNCHANGED)
        if heatmap_img is None:
            print(f"Failed to read heatmap image for player {player_id}")
            return None

        heatmap_img = cv2.resize(
            heatmap_img, (self.court_end_x - self.court_start_x, self.court_end_y - self.court_start_y)
        )

        # Add debug print
        print(f"Heatmap shape: {heatmap_img.shape}, dtype: {heatmap_img.dtype}")
        print(f"Heatmap min: {heatmap_img.min()}, max: {heatmap_img.max()}")  # Add this line

        return heatmap_img

    def draw_minicourt_with_heatmap(self, frame, player_id):
        # Draw the basic minicourt
        frame = self.draw_court(self.draw_bg_rectangle(frame))

        # Generate and overlay the heatmap
        heatmap_overlay = self.generate_heatmap_overlay(player_id)
        if heatmap_overlay is None:
            return frame

        # Create a mask for the court area
        mask = np.zeros((self.court_end_y - self.court_start_y, self.court_end_x - self.court_start_x), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (mask.shape[1], mask.shape[0]), 255, -1)

        # Apply the mask to the heatmap
        heatmap_overlay = cv2.bitwise_and(heatmap_overlay, heatmap_overlay, mask=mask)

        # Separate the color and alpha channels
        heatmap_bgr = heatmap_overlay[:, :, :3]
        heatmap_alpha = heatmap_overlay[:, :, 3]

        # Create a BGR image with the same dimensions as the court area
        heatmap_bgr_3channel = np.zeros(
            (self.court_end_y - self.court_start_y, self.court_end_x - self.court_start_x, 3), dtype=np.uint8
        )
        heatmap_bgr_3channel[:] = heatmap_bgr

        # Normalize alpha channel
        heatmap_alpha_normalized = heatmap_alpha.astype(float) / 255.0

        # Overlay the heatmap on the frame
        court_area = frame[self.court_start_y : self.court_end_y, self.court_start_x : self.court_end_x]

        # Use addWeighted for blending instead of alpha blending
        blended = cv2.addWeighted(court_area, 1, heatmap_bgr, 0.7, 0)

        # Apply the alpha channel
        alpha = np.expand_dims(heatmap_alpha_normalized, axis=2)
        court_area = (1 - alpha) * court_area + alpha * blended

        blending_factor = 0.7

        for c in range(3):
            court_area[:, :, c] = (
                court_area[:, :, c] * (1 - heatmap_alpha_normalized) + heatmap_bgr[:, :, c] * heatmap_alpha_normalized
            )

        for c in range(3):
            court_area[:, :, c] = (
                court_area[:, :, c] * (1 - blending_factor * heatmap_alpha_normalized)
                + heatmap_bgr[:, :, c] * blending_factor * heatmap_alpha_normalized
            )

        frame[self.court_start_y : self.court_end_y, self.court_start_x : self.court_end_x] = court_area.astype(
            np.uint8
        )

        # Add debug prints
        print(f"Court area shape: {court_area.shape}")
        print(f"Heatmap overlay shape: {heatmap_overlay.shape}")
        print(f"Frame shape: {frame.shape}")
        print(f"Heatmap overlay min: {heatmap_overlay.min()}, max: {heatmap_overlay.max()}")  # Add this line

        return frame

    def process_frames_with_heatmap(self, frames, player_positions):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            self.update_heatmap(player_positions[frame_num])

            # Draw heatmap for player 1
            frame_with_heatmap_1 = self.draw_minicourt_with_heatmap(frame.copy(), 1)

            # Draw heatmap for player 2
            frame_with_heatmap_2 = self.draw_minicourt_with_heatmap(frame.copy(), 2)

            # Combine both heatmaps
            combined_frame = cv2.addWeighted(frame_with_heatmap_1, 0.5, frame_with_heatmap_2, 0.5, 0)

            output_frames.append(combined_frame)

        return output_frames

    def draw_minicourt_with_heatmaps(self, frames, player_positions):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            self.update_heatmap(player_positions[frame_num])

            # Draw heatmap for player 1
            frame_with_heatmap_1 = self.draw_minicourt_with_heatmap(frame.copy(), 1)

            # Draw heatmap for player 2
            frame_with_heatmap_2 = self.draw_minicourt_with_heatmap(frame.copy(), 2)

            # Combine both heatmaps
            combined_frame = cv2.addWeighted(frame_with_heatmap_1, 0.5, frame_with_heatmap_2, 0.5, 0)

            output_frames.append(combined_frame)

        return output_frames

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
