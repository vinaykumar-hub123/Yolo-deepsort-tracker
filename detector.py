import numpy as np
from ultralytics import YOLO

class YOLODetector:
    """
    A wrapper class for YOLOv8 object detection using the Ultralytics library.
    """
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLOv8 model.
        
        Args:
            model_name (str): The pre-trained YOLOv8 model to load.
                             Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
                             The lightweight 'yolov8n.pt' (nano) is default for real-time performance.
        """
        print(f"[YOLO Detector] Loading pre-trained model: {model_name}...")
        self.model = YOLO(model_name)
        print("[YOLO Detector] Model loaded successfully.")

    def detect(self, frame, conf_threshold=0.25, iou_threshold=0.45, classes=None):
        """
        Performs object detection on a single frame.
        
        Args:
            frame (np.ndarray): The input image frame (OpenCV BGR format).
            conf_threshold (float): Confidence score threshold.
            iou_threshold (float): Intersection Over Union (IoU) threshold for NMS.
            classes (list): List of class IDs to filter detections (e.g., [0] for persons only).
            
        Returns:
            detections (np.ndarray): Array of detections of shape (N, 6)
                                    where each row is [x1, y1, x2, y2, confidence, class_id]
        """
        # Run inference using Ultralytics YOLOv8
        # verbose=False disables spamming console outputs on every frame
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            verbose=False
        )

        detections = []
        
        # Parse inference results
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Convert coordinates to numpy float list
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append([
                    float(xyxy[0]), # x1
                    float(xyxy[1]), # y1
                    float(xyxy[2]), # x2
                    float(xyxy[3]), # y2
                    conf,           # confidence score
                    cls             # class id
                ])
                
        return np.array(detections) if len(detections) > 0 else np.empty((0, 6))

    def get_class_name(self, class_id):
        """
        Retrieves the readable class name from a class ID.
        """
        return self.model.names.get(class_id, "Unknown")