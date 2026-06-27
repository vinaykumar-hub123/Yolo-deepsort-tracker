import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):
    """
    Computes Intersection Over Union (IoU) between two batches of bounding boxes.
    Used for cost matrix calculation in track association.
    """
    bb_test = np.expand_dims(bb_test, axis=1) # (N, 1, 4)
    bb_gt = np.expand_dims(bb_gt, axis=0)    # (1, M, 4)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    wh = w * h
    
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box [x1, y1, x2, y2] -> state z [x, y, s, r] (center, scale, ratio).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale (area)
    r = float(w) / float(h) if h != 0 else 0
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None, class_id=None):
    """
    Takes state x=[x,y,s,r,x_dot,y_dot,s_dot]^T -> bounding box [x1, y1, x2, y2].
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    
    bbox = [x1[0], y1[0], x2[0], y2[0]]
    if score is not None:
        bbox.append(score)
    if class_id is not None:
        bbox.append(class_id)
        
    return bbox

class KalmanBoxTracker:
    """
    Tracks state of a single target object using a 2D bounding box Kalman Filter.
    """
    count = 0
    def __init__(self, bbox, class_id=0):
        # State: [x, y, s, r, x_dot, y_dot, s_dot]^T (center x,y, scale, ratio, velocities)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = int(class_id)
        
    def update(self, bbox, class_id=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if class_id is not None:
            self.class_id = int(class_id)
        self.kf.update(convert_bbox_to_z(bbox))
        
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

class SortTracker:
    """
    Manages KalmanBoxTracker instances and coordinates association.
    """
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0:4]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        for index in sorted(to_del, reverse=True):
            self.trackers.pop(index)
            
        trks = np.delete(trks, to_del, axis=0)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(dets[det_idx, 0:4], class_id=dets[det_idx, 5])
            
        for i in unmatched_dets:
            if i < len(dets):
                trk = KalmanBoxTracker(dets[i, 0:4], class_id=dets[i, 5])
                self.trackers.append(trk)
                
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d[0:4], [trk.id, trk.class_id])).reshape(1, -1))
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
        iou_matrix = iou_batch(detections[:, 0:4], trackers[:, 0:4])
        if min(iou_matrix.shape) > 0:
            a, b = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack((a, b), axis=1)
        else:
            matched_indices = np.empty((0, 2), dtype=int)
            
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)