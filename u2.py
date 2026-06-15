import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import shlex
import torch

def start_store_analytics():
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到可用的 NVIDIA CUDA 环境！")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ 成功握手 NVIDIA 显卡: {gpu_name}！启动抗闪烁高级平滑引擎...")

    # 1. 加载模型
    model = YOLO("yolov8n.pt")
    
    # 2. 视频源与管道配置
    rtsp_url = "rtsp://Mikhail:999ookjdf@10.10.19.231/stream2"
    width, height = 1280, 720 
    
    ffmpeg_cmd = (
        f'ffmpeg -rtsp_transport tcp -i "{rtsp_url}" '
        f'-f image2pipe -pix_fmt bgr24 -vcodec rawvideo -an -sn -'
    )
    
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

    print("🔥 工业级抗闪烁引擎已就绪，按 'q' 键退出...")
    frame_size = width * height * 3

    with torch.inference_mode():
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()

            # 💡 优化一：5060 性能爆棚，我们【关闭跳帧】，每帧必检，从源头消灭因跳帧引起的闪烁
            # 💡 优化二：改用 tracker='bytetrack.yaml'，并把置信度 conf 提到 0.4（低于 0.4 的杂质直接过滤）
            # 💡 优化三：iou=0.5 降低重叠误判
            results = model.track(
                source=frame, 
                persist=True, 
                classes=[0], 
                device=0, 
                verbose=False, 
                imgsz=640,
                tracker="bytetrack.yaml",  # 换用工业级 ByteTrack 算法
                conf=0.4,                  # 提高置信度阈值，防止高噪声干扰
                iou=0.5
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)
                    
                    # 绘制平滑绿框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Customer ID: {track_id}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if track_id not in already_counted_ids:
                        already_counted_ids.add(track_id)
                        total_customer_count += 1
                    
                    # 降低每一帧的增量（从 1.2 降到 0.3），让热力图呈线性丝滑渲染，绝不突变
                    cv2.circle(accum_heatmap, (cx, cy), radius=20, color=(0.3), thickness=-1)

            # 4. 💡 优化四：平滑热力图渲染（告别动态归一化闪烁）
            if np.any(accum_heatmap):
                blur_heatmap = cv2.GaussianBlur(accum_heatmap, (51, 51), 0)
                
                # 【核心修改】放弃全局动态缩放，改用固定上限阈值（比如停留 200 帧判定为极度火热）
                # 这样即使画面里的人消失了，热力图的颜色也不会产生剧烈反差闪烁
                heatmap_fixed = np.clip(blur_heatmap * 5, 0, 255).astype(np.uint8)
                
                color_heatmap = cv2.applyColorMap(heatmap_fixed, cv2.COLORMAP_JET)
                overlay_frame = cv2.addWeighted(frame, 0.7, color_heatmap, 0.3, 0)
            else:
                overlay_frame = frame

            # 5. BI 数据面板
            cv2.rectangle(overlay_frame, (10, 10), (280, 50), (0, 0, 0), -1)
            cv2.putText(overlay_frame, f"Total Customers: {total_customer_count}", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 6. 显示窗口
            cv2.imshow("Store BI - Smooth NVIDIA CUDA Engine", overlay_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_store_analytics()