from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_pos(self, ball_pos):
        # Extract ball positions or use empty lists if no ball is detected
        ball_pos = [x.get(1, []) for x in ball_pos]

        # Convert list to pandas dataframe
        df_ball_pos = pd.DataFrame(ball_pos, columns=["x1", "y1", "x2", "y2"])

        # Create a time array
        time = np.arange(len(df_ball_pos))

        # Interpolate each coordinate separately using cubic spline interpolation
        for col in df_ball_pos.columns:
            # Find non-NaN values
            mask = ~np.isnan(df_ball_pos[col])
            if np.sum(mask) > 3:  # Need at least 4 points for cubic interpolation
                interp_func = interp1d(time[mask], df_ball_pos[col][mask], kind="cubic", fill_value="extrapolate")
                df_ball_pos[col] = interp_func(time)
            else:
                # Fall back to linear interpolation if we have fewer than 4 points
                interp_func = interp1d(time[mask], df_ball_pos[col][mask], kind="linear", fill_value="extrapolate")
                df_ball_pos[col] = interp_func(time)

        # Round the interpolated values to integers
        df_ball_pos = df_ball_pos.round().astype(int)

        # Convert back to the original format
        ball_pos = [{1: x} for x in df_ball_pos.to_numpy().tolist()]

        return ball_pos

    def get_ball_hit_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        df_ball_positions["ball_hit"] = 0

        df_ball_positions["mid_y"] = (df_ball_positions["y1"] + df_ball_positions["y2"]) / 2
        df_ball_positions["mid_y_rolling_mean"] = (
            df_ball_positions["mid_y"].rolling(window=15, min_periods=1, center=False).mean()
        )
        df_ball_positions["delta_y"] = df_ball_positions["mid_y_rolling_mean"].diff()
        minimum_change_frames_for_hit = 10

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = (
                df_ball_positions["delta_y"].iloc[i] > 0 and df_ball_positions["delta_y"].iloc[i + 1] < 0
            )
            positive_position_change = (
                df_ball_positions["delta_y"].iloc[i] < 0 and df_ball_positions["delta_y"].iloc[i + 1] > 0
            )

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] > 0 and df_ball_positions["delta_y"].iloc[change_frame] < 0
                    )
                    positive_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] < 0 and df_ball_positions["delta_y"].iloc[change_frame] > 0
                    )

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions["ball_hit"].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions["ball_hit"] == 1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames):
        ball_detections = []

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        return ball_detections

    def detect_frames_from_stub(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames

    def draw_trajectory(self, frames, ball_detections, window_size=30, start_delay=10):
        trajectory_frames = []

        for i, (frame, ball_dict) in enumerate(zip(frames, ball_detections)):
            frame_copy = frame.copy()

            if i >= start_delay:
                # Get the ball positions for the last 'window_size' frames
                start_index = max(start_delay, i - window_size)
                recent_detections = ball_detections[start_index : i + 1]

                # Draw the trajectory
                for j, detection in enumerate(reversed(recent_detections)):
                    if 1 in detection and len(detection[1]) >= 4:
                        x1, y1, x2, y2 = map(int, detection[1])

                        # Calculate center of the ball
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                        # Calculate the ball's radius
                        radius = max(2, (x2 - x1) // 2)  # Ensure a minimum size

                        # Draw a solid yellow circle with varying opacity
                        color = (0, 255, 255)  # Yellow color
                        opacity = 0.7 * (j + 1) / len(recent_detections)

                        overlay = frame_copy.copy()
                        cv2.circle(overlay, (center_x, center_y), radius, color, -1)  # -1 fills the circle
                        cv2.addWeighted(overlay, opacity, frame_copy, 1 - opacity, 0, frame_copy)

            # Always draw the current ball position
            if 1 in ball_dict and len(ball_dict[1]) >= 4:
                x1, y1, x2, y2 = map(int, ball_dict[1])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(2, (x2 - x1) // 2)
                cv2.circle(frame_copy, (center_x, center_y), radius, (0, 255, 255), -1)  # Solid yellow circle

            trajectory_frames.append(frame_copy)

        return trajectory_frames

    def overlay_ball(self, background, ball, x, y, opacity):
        """Overlay the ball image on the background with given opacity."""
        if ball.shape[2] == 3:
            ball = cv2.cvtColor(ball, cv2.COLOR_BGR2BGRA)

        # Create a mask of the ball
        mask = cv2.threshold(ball[:, :, 3], 0, 255, cv2.THRESH_BINARY)[1]

        # Create the overlay image with the desired opacity
        overlay = np.zeros(background.shape, dtype=np.uint8)
        overlay[y : y + ball.shape[0], x : x + ball.shape[1]] = ball[:, :, :3]

        # Blend the overlay with the background
        cv2.addWeighted(overlay, opacity, background, 1 - opacity, 0, background)

        # Add the ball to the background using the mask
        roi = background[y : y + ball.shape[0], x : x + ball.shape[1]]
        result = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        result += cv2.bitwise_and(
            overlay[y : y + ball.shape[0], x : x + ball.shape[1]],
            overlay[y : y + ball.shape[0], x : x + ball.shape[1]],
            mask=mask,
        )
        background[y : y + ball.shape[0], x : x + ball.shape[1]] = result

        return background
