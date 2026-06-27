import os
import sys
import argparse
import cv2
import numpy as np

from detector import YOLODetector
from tracker import SortTracker
from utils import FPSCounter, draw_tracks, draw_hud

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Multi-Object Detection and Tracking with YOLO & SORT")
    parser.add_argument("--source", type=str, default="0", help="Path to video file, or '0' for web camera live capture feed.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model (yolov8n.pt, yolov8s.pt, etc.).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence score filtering threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for non-maximum suppression.")
    parser.add_argument("--max-age", type=int, default=15, help="Max consecutive frames to retain lost tracks.")
    parser.add_argument("--min-hits", type=int, default=3, help="Min consecutive frames to declare a track active.")
    parser.add_argument("--tracker-iou", type=float, default=0.3, help="Min IoU overlap to match detections to tracks.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save processed output video.")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without spawning the cv2.imshow GUI.")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("         YOLO & SORT MULTI-OBJECT TRACKING ENGINE")
    print("=" * 60)
    
    try:
        detector = YOLODetector(model_name=args.model)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load YOLOv8 model: {e}")
        sys.exit(1)
        
    tracker = SortTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.tracker_iou)
    source = args.source
    if source.isdigit():
        source = int(source)
        print(f"[Video Source] Initializing Web Camera Index: {source}...")
    else:
        if not os.path.exists(source):
            print(f"[CRITICAL ERROR] Video file not found: {source}")
            sys.exit(1)
        print(f"[Video Source] Loading Video File: {source}...")
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[CRITICAL ERROR] Unable to open video source feed.")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Video Specs] Resolution: {width}x{height} | Source FPS: {fps_src:.2f}")
    
    writer = None
    if args.output:
        print(f"[Video Writer] Writing outputs to: {args.output}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps_src if fps_src > 0 else 30.0
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width, height))
        
    fps_counter = FPSCounter()
    trail_history = {}
    print("[Pipeline Status] Processing started. Press 'q' inside visual window to exit.")
    print("-" * 60)
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Pipeline Event] Video source stream completed.")
                break
                
            frame_idx += 1
            current_fps = fps_counter.tick()
            
            # STEP A: Detection
            detections = detector.detect(frame, conf_threshold=args.conf, iou_threshold=args.iou)
            
            # STEP B: Tracking
            tracks = tracker.update(detections)
            
            # STEP C: Drawing Overlay & Trails
            draw_tracks(frame, tracks, detector.get_class_name, trail_history)
            
            # STEP D: Drawing HUD panel
            draw_hud(frame, current_fps, len(tracks), {'conf': args.conf, 'max_age': args.max_age})
            
            if writer:
                writer.write(frame)
                
            if not args.no_gui:
                cv2.imshow("YOLO & Deep SORT Real-time Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Pipeline Event] Interrupted by user key.")
                    break
                    
            if frame_idx % 30 == 0:
                print(f"Frame: {frame_idx:04d} | Live FPS: {current_fps:.1f} | Active Tracks: {len(tracks)}")
                
    except KeyboardInterrupt:
        print("\n[Pipeline Event] Process terminated by system keyboard interrupt.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print("=" * 60)
        print(f"Pipeline Terminated. Total Processed Frames: {frame_idx}")
        print("=" * 60)

if __name__ == "__main__":
    main()