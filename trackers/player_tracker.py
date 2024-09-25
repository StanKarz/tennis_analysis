from utils import measure_distance, get_center_bbox
from ultralytics import YOLO
import cv2
import pickle
import sys

sys.path.append("../")


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # Get detections for first frame to choose players
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)

        # Filter detections to keep only the chosen players
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player
            }
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []

        # Calculate the distane of each player to the court keypoints
        for track_id, bbox in player_dict.items():
            player_center = get_center_bbox(bbox)

            min_distance = float("inf")
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort the players by their distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Log distances for debugging
        print(f"Player distances to court keypoints:{distances}")

        # Choose the first two players closest to the court
        if len(distances) >= 2:
            chosen_players = [distances[0][0], distances[1][0]]
        else:
            print(f"Warning only {len(distances)} players detected. Expected 2 players")
            chosen_players = {distances[i][0] for i in range(len(distances))}

        print("Chosen player IDs: {chosen_players}")
        return chosen_players

    def detect_frames(self, frames):
        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        return player_detections

    def detect_frames_from_stub(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # If reading from stub, load detections from the file
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        # Otherwise detect players in each frame
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # Save detections to a stub file
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]  # Run YOLO tracking on the frame
        id_name_dict = results.names  # Map of object class IDs to names

        player_dict = {}
        # Process each detected object and filter for "person"
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])  # Get tracking ID of the object
            result = box.xyxy.tolist()[0]  # Get the bounding box coordinates
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        # Iterate over each frame and corresponding player detections
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes for each detected player
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox

                # Draw player ID above the bounding box
                cv2.putText(
                    frame,
                    f"Player ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                # Draw bounding box around player
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
