"""Batch-one YOLOv8/11 detect ONNX, raw [1, 4+nc, N], nms=False."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:
    def __init__(self, model, size=640, confidence=0.25, iou=0.45,
                 classes=(0,), num_classes=80, threads=1):
        if not 0 < confidence < 1 or not 0 < iou < 1 or threads < 1:
            raise ValueError("Invalid confidence, IoU, or thread count")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(Path(model)), options,
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        if inp.type != "tensor(float)" or len(inp.shape) != 4:
            raise ValueError("Expected float32 NCHW detect model")
        if isinstance(inp.shape[0], int) and inp.shape[0] != 1:
            raise ValueError("Only batch-one models supported")
        self.height = inp.shape[2] if isinstance(inp.shape[2], int) else size
        self.width = inp.shape[3] if isinstance(inp.shape[3], int) else size
        self.input_name = inp.name
        self.confidence, self.iou = confidence, iou
        self.classes, self.nc = classes, num_classes

    def __call__(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (self.width - nw) // 2, (self.height - nh) // 2
        canvas = np.full((self.height, self.width, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None],
                                     dtype=np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: tensor})[0]
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4 + self.nc:
            raise ValueError(f"Expected [1,{4+self.nc},N] raw detection output; got {raw.shape}")
        pred = raw[0].T
        cls = pred[:, 4:].argmax(axis=1)
        scores = pred[np.arange(len(pred)), cls + 4]
        keep = np.isfinite(pred).all(axis=1) & (scores >= self.confidence)
        if self.classes is not None:
            keep &= np.isin(cls, self.classes)
        pred, cls, scores = pred[keep], cls[keep], scores[keep]
        boxes = pred[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2  # xywh for OpenCV NMS
        selected = []
        for label in np.unique(cls):
            indices = np.flatnonzero(cls == label)
            local = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices].tolist(),
                                     self.confidence, self.iou)
            selected.extend(indices[np.asarray(local, dtype=int).reshape(-1)].tolist())
        selected = sorted(selected, key=lambda i: float(scores[i]), reverse=True)[:300]
        boxes = boxes[selected]
        boxes[:, 2:] += boxes[:, :2]  # xyxy
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        result = np.column_stack((boxes, scores[selected], cls[selected])).astype(np.float32)
        return result[(result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])]
