from utils import (read_video,
                   save_video)

from trackers import (PlayerTracker)


def main():
    # process video
    input_path = "input/vid_1.mp4"
    video_frames = read_video(input_path)

    # detecting players
    player_tracker = PlayerTracker(model_path="models/yolo8x.pt")
    player_detections = player_tracker.detect_frames(video_frames)

    save_video(video_frames, "output/vid_1.avi")


if __name__ == "__main__":
    main()
