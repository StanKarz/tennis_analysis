from utils import (read_video,
                   save_video)
from trackers import (PlayerTracker, BallTracker)
from court_kp_detector import CourtKPDetector
from mini_court import MiniCourt
import cv2


def main():
    # process video
    input_path = "input/input_video.mp4"
    video_frames = read_video(input_path)

    # detecting players + ball

    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    ball_tracker = BallTracker(model_path="models/yolov8_trained.pt")

    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')

    # detecting balls
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')

    ball_detections = ball_tracker.interpolate_ball_pos(ball_detections)

    # Detecting court keypoints
    court_model_path = 'models/keypoints_model_100resnet.pth'
    court_kp_detector = CourtKPDetector(model_path=court_model_path)
    court_kps = court_kp_detector.predict(video_frames[0])

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(
        court_kps, player_detections)

    # Init MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(
        video_frames, player_detections)

    # Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(
        video_frames, ball_detections)

    # TODO: Draw court keypoints (TRY A BETTER METHOD FOR INTERPOLATION)
    output_video_frames = court_kp_detector.draw_keypoints_on_video(
        output_video_frames, court_kps)

    # draw MiniCourt
    output_video_frames = mini_court.draw_minicourt(output_video_frames)

    # Draw frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    save_video(output_video_frames, "output/vid_TEST.avi")


if __name__ == "__main__":
    main()
