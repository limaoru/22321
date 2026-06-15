import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import shlex
import torch

def start_store_analytics():
    # 0. 🚀 检查 CUDA 是否在当前 Python 环境中可用
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到可用的 NVIDIA CUDA 环境！请确认是否正确安装了显卡驱动和 PyTorch-CUDA 版本。")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ 成功握手 NVIDIA 显卡: {gpu_name}！准备火力全开...")

    # 1. 加载轻量级 Nano 模型 (5060 跑它延迟一般在 1-3 毫秒左右)
    print("⏳ 正在加载 YOLOv8-Nano 模型...")
    model = YOLO("yolov8n.pt")
    
    # 2. 视频源与管道配置（继续使用稳定高效的系统 FFmpeg 管道）
    rtsp_url = "rtsp://Mikhail:999ookjdf@10.10.19.231/stream2"
    width, height = 1280, 720 
    
    ffmpeg_cmd = (
        f'ffmpeg -rtsp_transport tcp -i "{rtsp_url}" '
        f'-f image2pipe -pix_fmt bgr24 -vcodec rawvideo -an -sn -'
    )
    
    print("🚀 正在通过系统管道拉起 FFmpeg 进程...")
    process = subprocess.Popen(
        shlex.split(ffmpeg_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=width * height * 3
    )

    # 3. 创建热力图画布与统计变量
    accum_heatmap = np.zeros((height, width), dtype=np.float32)
    total_customer_count = 0
    already_counted_ids = set()
    
    frame_counter = 0
    detect_every_n_frames = 2  # 每 2 帧推理一次，兼顾极致流畅度与极低显卡功耗

    print("🔥 CUDA 推理引擎已成功挂载！开始实时客流分析，按 'q' 键退出...")
    frame_size = width * height * 3

    # 使用 inference_mode 彻底关闭梯度计算，压榨显卡吞吐量
    with torch.inference_mode():
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                print("⚠️ 警告：管道流中断。")
                break

            frame_counter += 1
            # 将原始字节流转化为可写的 numpy 矩阵副本
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()

            # 隔帧触发 YOLO 推理
            if frame_counter % detect_every_n_frames == 0:
                # 💡 核心修改：device=0 强制走英伟达显卡，imgsz 恢复到标准 640 保证高精度
                results = model.track(source=frame, persist=True, classes=[0], device=0, verbose=False, imgsz=640)
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2)
                        cy = int(y2)
                        
                        # 绘制当前人的绿框
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 客流计数
                        if track_id not in already_counted_ids:
                            already_counted_ids.add(track_id)
                            total_customer_count += 1
                        
                        # 热力图能量累加
                        cv2.circle(accum_heatmap, (cx, cy), radius=18, color=(1.2), thickness=-1)

            # 4. 极速渲染热力图
            if np.any(accum_heatmap):
                blur_heatmap = cv2.GaussianBlur(accum_heatmap, (31, 31), 0)
                heatmap_normalized = cv2.normalize(blur_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                color_heatmap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
                overlay_frame = cv2.addWeighted(frame, 0.7, color_heatmap, 0.3, 0)
            else:
                overlay_frame = frame

            # 5. 渲染 BI 数据面板
            cv2.rectangle(overlay_frame, (10, 10), (280, 50), (0, 0, 0), -1)
            cv2.putText(overlay_frame, f"Total Customers: {total_customer_count}", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 6. 弹出标准窗口
            cv2.imshow("Store Analytics - NVIDIA 5060 CUDA Engine", overlay_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_store_analytics()