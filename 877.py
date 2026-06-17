import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import threading
import time
from queue import Queue

# ==================== 🎛️ 监控流配置中心 ====================
CAMERA_CONFIGS = {
    "Cam_231_Counter": {
        "rtsp": "rtsp://Mikhail:999ookjdf@10.10.19.231/stream1",
        "width": 2304,
        "height": 1296
    },
    "Cam_104_Specialty": {
        "rtsp": "rtsp://Mikhail:999ookjdf@10.10.19.104/stream1",
        "width": 2304,
        "height": 1296
    }
}

# 🎯 狠活配置：我们要捕获的目标 COCO 类别 ID
# 0: person, 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 41: cup, 67: cell phone
TARGET_CLASSES = [0, 24, 26, 28, 39, 41, 67]

CLASS_MAPPING = {
    0: ("Person", (0, 255, 0)),        # 绿色
    24: ("Backpack", (255, 165, 0)),   # 橙色
    26: ("Handbag", (255, 0, 255)),    # 紫色
    28: ("Suitcase", (0, 165, 255)),   # 琥珀色
    39: ("Bottle", (0, 255, 255)),     # 黄色
    41: ("Cup", (0, 128, 255)),        # 浅橙
    67: ("Phone", (0, 0, 255))         # 红色
}

# COCO 17 关键点骨架连接
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
# =========================================================

def draw_skeleton(frame, keypoints_xy, keypoints_conf=None, color=(0, 255, 0), thickness=2, conf_thresh=0.5):
    """在帧上绘制人体骨架与关键点"""
    if keypoints_xy is None:
        return
    kps = keypoints_xy.cpu().numpy() if hasattr(keypoints_xy, "cpu") else np.asarray(keypoints_xy)
    confs = None
    if keypoints_conf is not None:
        confs = keypoints_conf.cpu().numpy() if hasattr(keypoints_conf, "cpu") else np.asarray(keypoints_conf)

    for p_idx, kp in enumerate(kps):
        kp_conf = confs[p_idx] if confs is not None else None
        for i, j in SKELETON:
            if kp[i][0] <= 0 or kp[i][1] <= 0 or kp[j][0] <= 0 or kp[j][1] <= 0:
                continue
            if kp_conf is not None and (kp_conf[i] < conf_thresh or kp_conf[j] < conf_thresh):
                continue
            cv2.line(frame, (int(kp[i][0]), int(kp[i][1])), (int(kp[j][0]), int(kp[j][1])),
                     color, thickness, cv2.LINE_AA)
        for x, y in kp:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def draw_person_contour(frame, mask_xy, color, thickness=2, fill_alpha=0.22):
    """用分割掩码多边形绘制人体轮廓（替代方框）"""
    if mask_xy is None or len(mask_xy) < 3:
        return None
    pts = np.asarray(mask_xy, dtype=np.int32).reshape(-1, 1, 2)
    if fill_alpha > 0:
        fill_layer = frame.copy()
        cv2.fillPoly(fill_layer, [pts], color)
        cv2.addWeighted(fill_layer, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    top_y = int(np.min(pts[:, 0, 1]))
    top_x = int(np.mean(pts[:, 0, 0]))
    return top_x, top_y


def build_heatmap_views(frame, accum_heatmap, width):
    """生成主画面热力叠加层与独立热力图窗口画面"""
    if not np.any(accum_heatmap):
        return frame.copy(), np.zeros_like(frame)

    blur_k = max(51, int(51 * width / 1280) | 1)
    blur_heatmap = cv2.GaussianBlur(accum_heatmap, (blur_k, blur_k), 0)
    heatmap_fixed = np.clip(blur_heatmap * 4, 0, 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_fixed, cv2.COLORMAP_JET)
    overlay_frame = cv2.addWeighted(frame, 0.65, color_heatmap, 0.35, 0)
    heatmap_panel = cv2.addWeighted(frame, 0.12, color_heatmap, 0.88, 0)
    return overlay_frame, heatmap_panel

class RTSPStreamReader(threading.Thread):
    def __init__(self, name, rtsp_url, width, height):
        super().__init__()
        self.name = name
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_queue = Queue(maxsize=3)
        self.stopped = False
        self.cap = None

    def _open_capture(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        print(f"📡 [线程启动] 正在通过 OpenCV 连接 {self.name} ...")
        self.cap = self._open_capture()

        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(2)
                self.cap = self._open_capture()
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                time.sleep(2)
                self.cap = self._open_capture()
                continue

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Queue.Empty:
                    pass
            self.frame_queue.put(frame)

        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.stopped = True

def start_advanced_store_analytics():
    if not torch.cuda.is_available():
        print("❌ 未检测到可用的 NVIDIA CUDA 环境！")
        return
    
    print("🔥 5060 多维度零售视界系统启动！加载 YOLOv8m-seg + 姿态骨架...")
    model = YOLO("yolov8m-seg.pt")
    pose_model = YOLO("yolo11n-pose.pt")

    readers = {}
    heatmaps = {}
    customer_counts = {}
    counted_ids = {}

    for cam_name, config in CAMERA_CONFIGS.items():
        cv2.namedWindow(f"Store Analytics Platform - {cam_name}", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"Heatmap - {cam_name}", cv2.WINDOW_NORMAL)
        reader = RTSPStreamReader(cam_name, config["rtsp"], config["width"], config["height"])
        reader.daemon = True
        reader.start()
        readers[cam_name] = reader
        
        heatmaps[cam_name] = np.zeros((config["height"], config["width"]), dtype=np.float32)
        customer_counts[cam_name] = 0
        counted_ids[cam_name] = set()

    with torch.inference_mode():
        while True:
            for cam_name, reader in readers.items():
                if reader.frame_queue.empty():
                    continue
                
                frame = reader.frame_queue.get()
                accum_heatmap = heatmaps[cam_name]
                frame_w = reader.width
                
                # 实时动态特征计数看板
                current_metrics = {name: 0 for name, _ in CLASS_MAPPING.values()}

                # 分割模型：人体输出掩码轮廓，其余类别仍用检测框
                results = model.track(
                    source=frame, persist=True, device=0, verbose=False, imgsz=640,
                    classes=TARGET_CLASSES, tracker="bytetrack.yaml", conf=0.35, iou=0.4
                )
                
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(clss)
                    masks_xy = results[0].masks.xy if results[0].masks is not None else None
                    
                    for idx, (box, cls_id, track_id) in enumerate(zip(boxes, clss, track_ids)):
                        if cls_id not in CLASS_MAPPING:
                            continue
                        
                        label, color = CLASS_MAPPING[cls_id]
                        current_metrics[label] += 1
                        
                        x1, y1, x2, y2 = box
                        id_str = f" ID:{track_id}" if track_id is not None else ""

                        if cls_id == 0:
                            mask_xy = masks_xy[idx] if masks_xy is not None and idx < len(masks_xy) else None
                            label_pos = draw_person_contour(frame, mask_xy, color)
                            if label_pos is None:
                                label_pos = (int(x1), int(y1))
                            cv2.putText(frame, f"{label}{id_str}", (label_pos[0], label_pos[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"{label}{id_str}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # 客流计数逻辑 (仅针对人类客流进行 ID 去重计数)
                        if cls_id == 0 and track_id not in counted_ids[cam_name] and track_id is not None:
                            counted_ids[cam_name].add(track_id)
                            customer_counts[cam_name] += 1
                        
                        # 客流热力图能量累加（以人脚底为中心）
                        if cls_id == 0:
                            cx, cy = int((x1 + x2) / 2), int(y2)
                            radius = max(22, int(22 * frame_w / 1280))
                            cv2.circle(accum_heatmap, (cx, cy), radius=radius, color=0.4, thickness=-1)

                # 人体姿态骨架叠加（仅当画面中存在行人时推理）
                if current_metrics["Person"] > 0:
                    pose_results = pose_model(frame, verbose=False, device=0, conf=0.35)[0]
                    if pose_results.keypoints is not None:
                        draw_skeleton(
                            frame,
                            pose_results.keypoints.xy,
                            pose_results.keypoints.conf,
                            color=CLASS_MAPPING[0][1],
                        )

                # 热力图：主画面叠加 + 独立热力图窗口
                overlay_frame, heatmap_panel = build_heatmap_views(frame, accum_heatmap, frame_w)

                # 📊 极速渲染多维度数据透明面板
                panel_h = 40 + len(current_metrics) * 25
                cv2.rectangle(overlay_frame, (10, 10), (320, panel_h), (0, 0, 0), -1)
                
                # 打印累计客流
                cv2.putText(overlay_frame, f"Total Customers: {customer_counts[cam_name]}", (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 动态打印当前帧的各种物品/属性统计
                for idx, (label, count) in enumerate(current_metrics.items()):
                    y_pos = 65 + idx * 25
                    _, color = next(v for k, v in CLASS_MAPPING.items() if v[0] == label)
                    cv2.putText(overlay_frame, f"Current {label}s: {count}", (20, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imshow(f"Store Analytics Platform - {cam_name}", overlay_frame)
                cv2.imshow(f"Heatmap - {cam_name}", heatmap_panel)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for reader in readers.values(): reader.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_advanced_store_analytics()