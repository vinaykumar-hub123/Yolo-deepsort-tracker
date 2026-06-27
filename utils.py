import cv2
import time
import numpy as np

class FPSCounter:
    """Measures and smooths the Frames Per Second (FPS) of the video pipeline."""
    def __init__(self, avg_window=30):
        self.avg_window = avg_window
        self.times = []
        self.fps = 0.0
        
    def tick(self):
        current_time = time.time()
        self.times.append(current_time)
        if len(self.times) > self.avg_window:
            self.times.pop(0)
        if len(self.times) > 1:
            time_diff = self.times[-1] - self.times[0]
            self.fps = (len(self.times) - 1) / time_diff if time_diff > 0 else 0.0
        return self.fps

def get_color(class_id, track_id=None):
    seed_val = int(class_id) if track_id is None else int(track_id)
    palette = [
        (255, 99, 71), (46, 139, 87), (30, 144, 255), (218, 112, 214),
        (255, 165, 0), (72, 61, 139), (0, 128, 128), (139, 69, 19),
        (186, 85, 211), (0, 206, 209)
    ]
    return palette[seed_val % len(palette)]

def draw_tracks(frame, tracks, class_names_map, trail_history=None):
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        class_id = int(class_id)
        
        color = get_color(class_id, track_id)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        if trail_history is not None:
            if track_id not in trail_history:
                trail_history[track_id] = []
            trail_history[track_id].append((cx, cy))
            if len(trail_history[track_id]) > 20:
                trail_history[track_id].pop(0)
            trail_points = trail_history[track_id]
            for i in range(1, len(trail_points)):
                thickness = int(np.sqrt(20 / float(i + 1)) * 2)
                cv2.line(frame, trail_points[i - 1], trail_points[i], color, thickness)
                
        class_name = class_names_map(class_id) if callable(class_names_map) else class_names_map.get(class_id, f"Cls {class_id}")
        label = f"ID {track_id} | {class_name}"
        
        # Double outline rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 1)
        cv2.rectangle(frame, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), color, 2)
        
        # Draw background textbox label
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_y = max(y1, text_height + 10)
        cv2.rectangle(frame, (x1, text_y - text_height - 6), (x1 + text_width + 10, text_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 5, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Stylish overlay glowing corners
        corner_len = min(15, int((x2 - x1) / 4), int((y2 - y1) / 4))
        if corner_len > 3:
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

def draw_hud(frame, fps, active_count, tracker_params=None):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (220, 110), (15, 15, 15), -1)
    
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, "YOLO & SORT HUD", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (20, 38), (210, 38), (100, 100, 100), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active Tracks: {active_count}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    if tracker_params:
        conf = tracker_params.get('conf', 0.25)
        max_age = tracker_params.get('max_age', 15)
        cv2.putText(frame, f"Conf: {conf:.2f} | MaxAge: {max_age}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)