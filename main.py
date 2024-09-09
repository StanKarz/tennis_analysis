from utils import (read_video,
                   save_video)

from trackers import (PlayerTracker)


def main():
    input_path = "input/vid_1.mp4"
    video_frames = read_video(input_path)

    player_tracker = PlayerTracker("models/yolo8_trained.pt")

    save_video(video_frames, "output/vid_1.avi")


if __name__ == "__main__":
    main()
