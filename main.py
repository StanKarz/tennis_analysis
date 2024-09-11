from utils import (read_video,
                   save_video)

from trackers import (PlayerTracker, BallTracker)


def main():
    # process video
    input_path = "input/vid_2.mp4"
    video_frames = read_video(input_path)

    # detecting players
    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detections.pkl')

    # detecting balls
    ball_tracker = BallTracker(model_path="models/yolov8_trained.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=False, stub_path='tracker_stubs/ball_detections.pkl')

    output_video_frames = player_tracker.draw_bboxes(
        video_frames, player_detections)

    output_video_frames = ball_tracker.draw_bboxes(
        video_frames, ball_detections)

    save_video(output_video_frames, "output/vid_2.avi")


if __name__ == "__main__":
    main()
