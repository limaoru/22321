import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import shlex
import torch
import threading
import time
from queue import Queue

# ==================== 🎛️ 监控流配置中心 ====================
CAMERA_CONFIGS = {
    "Cam_231_Counter": {
        "rtsp": "rtsp://Mikhail:999ookjdf@10.10.19.231/stream2",
        "width": 1280,
        "height": 720
    },
    "Cam_104_Specialty": {
        # 💡 请在此处修正 19.104 摄像头的实际 RTSP 账号、密码及后缀路径
        "rtsp": "rtsp://admin:admin12345@10.10.19.104/h264/ch1/sub/av_stream",
        "width": 1280,
        "height": 720
    }
}
# =========================================================

class RTSPStreamReader(threading.Thread):
    """高级异步拉流器：利用独立线程死磕 FFmpeg 管道，确保主显卡永远只读到最新鲜的帧"""
    def __init__(self, name, rtsp_url, width, height):
        super().__init__()
        self.name = name
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.frame_queue = Queue(maxsize=3) # 严格控制缓冲区，宁可丢帧也绝不积压延迟
        self.stopped = False
        
        self.ffmpeg_cmd = (
            f'ffmpeg -rtsp_transport tcp -i "{self.rtsp_url}" '
            f'-f image2pipe -pix_fmt bgr24 -vcodec rawvideo -an -sn -'
        )

    def run(self):
        print(f"📡 [线程启动] 正在为 {self.name} 拉起异步 FFmpeg 管道...")
        process = subprocess.Popen(
            shlex.split(self.ffmpeg_cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size
        )

        while not self.stopped:
            raw_frame = process.stdout.read(self.frame_size)
            if len(raw_frame) != self.frame_size:
                print(f"⚠️  [警告] {self.name} 管道流断开，正在尝试重启...")
                process.terminate()
                time.sleep(2)
                process = subprocess.Popen(shlex.split(self.ffmpeg_cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.frame_size)
                continue

            # 将原始字节转化为 NumPy 矩阵
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()

            # 如果队列满了，强行弹出一个旧帧（弃子策略），塞入最新帧，将实时延迟干到 0 毫秒
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Queue.Empty:
                    pass
            self.frame_queue.put(frame)

        process.terminate()

    def stop(self):
        self.stopped = True

def start_multi_stream_analytics():
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到可用的 NVIDIA CUDA 环境！")
        return
    
    print(f"🔥 5060 双路全功率并发引擎启动！加载顶级大模型 YOLOv8x...")
    model = YOLO("yolov8x.pt")

    # 1. 启动所有摄像头的异步拉流线程
    readers = {}
    heatmaps = {}
    customer_counts = {}
    counted_ids = {}

    for cam_name, config in CAMERA_CONFIGS.items():
        reader = RTSPStreamReader(cam_name, config["rtsp"], config["width"], config["height"])
        reader.daemon = True
        reader.start()
        readers[cam_name] = reader
        
        # 为每路流初始化独立型 BI 统计变量
        heatmaps[cam_name] = np.zeros((config["height"], config["width"]), dtype=np.float32)
        customer_counts[cam_name] = 0
        counted_ids[cam_name] = set()

    print("🚀 所有视频流线程已挂载！主线程显卡准备就绪。按 'q' 键退出...")

    with torch.inference_mode():
        while True:
            for cam_name, reader in readers.items():
                # 如果这个摄像头当前还没吐出第一帧，先跳过等一等
                if reader.frame_queue.empty():
                    continue
                
                frame = reader.frame_queue.get()
                accum_heatmap = heatmaps[cam_name]

                # 2. 5060 显卡全速推理单帧
                results = model.track(
                    source=frame, 
                    persist=True, 
                    classes=[0], 
                    device=0, 
                    verbose=False, 
                    imgsz=1280,                # 1280原画点对点检测，极强抗干扰
                    tracker="bytetrack.yaml",  # 标配工业级 ByteTrack 算法
                    conf=0.45,
                    iou=0.4
                )
                
                # 3. 业务逻辑：框选、计数与热力图累加
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2)
                        cy = int(y2)
                        
                        # 绿框标记
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 每路流独立去重计数
                        if track_id not in counted_ids[cam_name]:
                            counted_ids[cam_name].add(track_id)
                            customer_counts[cam_name] += 1
                        
                        # 动态生成专属热力点
                        cv2.circle(accum_heatmap, (cx, cy), radius=22, color=(0.4), thickness=-1)

                # 4. 独立热力图渲染
                if np.any(accum_heatmap):
                    blur_heatmap = cv2.GaussianBlur(accum_heatmap, (51, 51), 0)
                    heatmap_fixed = np.clip(blur_heatmap * 4, 0, 255).astype(np.uint8)
                    color_heatmap = cv2.applyColorMap(heatmap_fixed, cv2.COLORMAP_JET)
                    overlay_frame = cv2.addWeighted(frame, 0.7, color_heatmap, 0.3, 0)
                else:
                    overlay_frame = frame

                # 5. 独立 BI 数据面板渲染
                cv2.rectangle(overlay_frame, (10, 10), (340, 50), (0, 0, 0), -1)
                cv2.putText(overlay_frame, f"{cam_name}: {customer_counts[cam_name]} Ppl", (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 6. 分别弹出独立窗口显示
                cv2.imshow(f"NVIDIA 5060 Engine - {cam_name}", overlay_frame)

            # 监听全局退出键 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 优雅释放所有子线程进程
    for reader in readers.values():
        reader.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_multi_stream_analytics()