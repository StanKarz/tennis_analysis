from utils import read_video, save_video, measure_distance, pixel_to_distance, draw_player_stats
from trackers import PlayerTracker, BallTracker
from court_kp_detector import CourtKPDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
import constants
from copy import deepcopy


def main():
    # process video
    input_path = "input/input_video.mp4"
    video_frames = read_video(input_path)

    # detecting players + ball

    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    ball_tracker = BallTracker(model_path="models/yolov8_trained.pt")

    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl"
    )

    # detecting balls
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl"
    )

    ball_detections = ball_tracker.interpolate_ball_pos(ball_detections)

    # Detecting court keypoints
    court_model_path = "models/resnet101_keypoints_model.pth"

    court_kp_detector = CourtKPDetector(model_path=court_model_path)
    court_kps = court_kp_detector.predict(video_frames[0])

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(court_kps, player_detections)

    # Init MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detect Ball Hits
    ball_hit_frames = ball_tracker.get_ball_hit_frames(ball_detections)
    print("Ball hit frames:", ball_hit_frames)

    # convert positions to minicourt positions
    player_minicourt_detections, ball_minicourt_detections = mini_court.convert_bb_to_minicourt_coordinates(
        player_detections, ball_detections, court_kps
    )

    player_stats_data = [
        {
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }
    ]

    for ball_shot_ind in range(len(ball_hit_frames) - 1):
        start_frame = ball_hit_frames[ball_shot_ind]
        end_frame = ball_hit_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_minicourt_detections[start_frame][1], ball_minicourt_detections[end_frame][1]
        )
        distance_covered_by_ball_meters = pixel_to_distance(
            distance_covered_by_ball_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_minicourt()
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_minicourt_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_minicourt_detections[start_frame][1]
            ),
        )

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_minicourt_detections[start_frame][opponent_player_id],
            player_minicourt_detections[end_frame][opponent_player_id],
        )
        distance_covered_by_opponent_meters = pixel_to_distance(
            distance_covered_by_opponent_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_minicourt()
        )

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = speed_of_ball_shot

        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += speed_of_opponent
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on="frame_num", how="left")
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df["player_1_average_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"] / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"] / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"] / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"] / player_stats_data_df["player_1_number_of_shots"]
    )

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    # TODO: Draw court keypoints (TRY A BETTER METHOD FOR INTERPOLATION)
    output_video_frames = court_kp_detector.draw_keypoints_on_video(output_video_frames, court_kps)

    # draw MiniCourt
    output_video_frames = mini_court.draw_minicourt(output_video_frames)
    output_video_frames = mini_court.draw_points_on_minicourt(output_video_frames, player_minicourt_detections)
    output_video_frames = mini_court.draw_points_on_minicourt(
        output_video_frames, ball_minicourt_detections, color=(0, 255, 255)
    )

    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Draw frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    save_video(output_video_frames, "output/TEST.avi")


if __name__ == "__main__":
    main()
