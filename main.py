import cv2
import pandas as pd
from utils import read_video, save_video, draw_player_stats, measure_distance, pixel_to_distance
from trackers import PlayerTracker, BallTracker
from court_kp_detector import CourtKPDetector
from mini_court import MiniCourt
import constants


def detect_players_and_ball(video_frames):
    """Detect players and ball in the video frames."""
    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    ball_tracker = BallTracker(model_path="models/yolov8_trained.pt")

    player_detections = player_tracker.detect_frames_from_stub(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl"
    )

    ball_detections = ball_tracker.detect_frames_from_stub(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl"
    )

    ball_detections = ball_tracker.interpolate_ball_pos(ball_detections)

    return player_detections, ball_detections, player_tracker, ball_tracker


def detect_court_keypoints(first_frame):
    """Detect court keypoints in the first frame (as they don't change throughout)."""
    court_model_path = "models/resnet101_keypoints_model.pth"
    court_kp_detector = CourtKPDetector(model_path=court_model_path)
    return court_kp_detector.predict(first_frame), court_kp_detector


def calculate_player_stats(
    player_minicourt_detections, ball_minicourt_detections, ball_hit_frames, mini_court, video_frames
):
    """Calculate player statistics based on ball hits and player movements."""
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
        start_frame, end_frame = ball_hit_frames[ball_shot_ind], ball_hit_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # Assuming 24fps

        # Calculate ball shot speed
        ball_distance = measure_distance(
            ball_minicourt_detections[start_frame][1], ball_minicourt_detections[end_frame][1]
        )
        ball_distance_meters = pixel_to_distance(
            ball_distance, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_minicourt()
        )
        ball_speed = ball_distance_meters / ball_shot_time_in_seconds * 3.6  # Convert to km/h

        # Determine which player hit the ball
        player_positions = player_minicourt_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_minicourt_detections[start_frame][1]
            ),
        )

        # Calculate opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        opponent_distance = measure_distance(
            player_minicourt_detections[start_frame][opponent_player_id],
            player_minicourt_detections[end_frame][opponent_player_id],
        )
        opponent_distance_meters = pixel_to_distance(
            opponent_distance, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_minicourt()
        )
        opponent_speed = opponent_distance_meters / ball_shot_time_in_seconds * 3.6  # Convert to km/h

        # Update player stats
        current_stats = player_stats_data[-1].copy()
        current_stats.update(
            {
                "frame_num": start_frame,
                f"player_{player_shot_ball}_number_of_shots": current_stats[
                    f"player_{player_shot_ball}_number_of_shots"
                ]
                + 1,
                f"player_{player_shot_ball}_total_shot_speed": current_stats[
                    f"player_{player_shot_ball}_total_shot_speed"
                ]
                + ball_speed,
                f"player_{player_shot_ball}_last_shot_speed": ball_speed,
                f"player_{opponent_player_id}_total_player_speed": current_stats[
                    f"player_{opponent_player_id}_total_player_speed"
                ]
                + opponent_speed,
                f"player_{opponent_player_id}_last_player_speed": opponent_speed,
            }
        )
        player_stats_data.append(current_stats)

    return process_player_stats(player_stats_data, len(video_frames))


def process_player_stats(player_stats_data, total_frames):
    """Process and calculate additional player statistics."""
    df = pd.DataFrame(player_stats_data)
    df = pd.merge(pd.DataFrame({"frame_num": range(total_frames)}), df, on="frame_num", how="left").ffill()

    for player in [1, 2]:
        df[f"player_{player}_average_shot_speed"] = (
            df[f"player_{player}_total_shot_speed"] / df[f"player_{player}_number_of_shots"]
        )
        df[f"player_{player}_average_player_speed"] = (
            df[f"player_{player}_total_player_speed"] / df[f"player_{3-player}_number_of_shots"]
        )

    return df


def draw_output_frames(
    video_frames,
    player_detections,
    ball_detections,
    court_kps,
    mini_court,
    player_minicourt_detections,
    ball_minicourt_detections,
    player_stats_df,
    player_tracker,
    ball_tracker,
    court_kp_detector,
):
    """Draw various elements on the output video frames."""
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_kp_detector.draw_keypoints_on_video(output_frames, court_kps)
    output_frames = mini_court.draw_minicourt(output_frames)
    output_frames = mini_court.draw_points_on_minicourt(output_frames, player_minicourt_detections)
    output_frames = mini_court.draw_points_on_minicourt(output_frames, ball_minicourt_detections, color=(0, 255, 255))
    output_frames = draw_player_stats(output_frames, player_stats_df)

    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return output_frames


def main():
    # Process video
    video_frames = read_video("input/input_video.mp4")

    # Detect players and ball
    player_detections, ball_detections, player_tracker, ball_tracker = detect_players_and_ball(video_frames)

    # Detect court keypoints
    court_kps, court_kp_detector = detect_court_keypoints(video_frames[0])

    # Choose and filter players
    player_detections = player_tracker.choose_and_filter_players(court_kps, player_detections)

    # Initialize MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detect ball hits
    ball_hit_frames = ball_tracker.get_ball_hit_frames(ball_detections)
    print("Ball hit frames:", ball_hit_frames)

    # Convert positions to minicourt coordinates
    player_minicourt_detections, ball_minicourt_detections = mini_court.convert_bb_to_minicourt_coordinates(
        player_detections, ball_detections, court_kps
    )

    # Calculate player stats
    player_stats_df = calculate_player_stats(
        player_minicourt_detections, ball_minicourt_detections, ball_hit_frames, mini_court, video_frames
    )

    # Draw output frames
    output_video_frames = draw_output_frames(
        video_frames,
        player_detections,
        ball_detections,
        court_kps,
        mini_court,
        player_minicourt_detections,
        ball_minicourt_detections,
        player_stats_df,
        player_tracker,
        ball_tracker,
        court_kp_detector,
    )

    # Save the output video
    save_video(output_video_frames, "output/analyzed_tennis_video.avi")


if __name__ == "__main__":
    main()
