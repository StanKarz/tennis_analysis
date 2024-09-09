from ultralytics import YOLO
import cv2


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)
        id_names_dict = results.names

        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            obj_cls_id = box.cls.tolist()[0]
            obj_cls_name = id_names_dict[obj_cls_id]
            if obj_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output_frames.append(frame)
