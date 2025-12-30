import cv2
import os
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
import logging
from collections import deque

# 默认日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AIAutoLabeler:
    """AI自动标注工具类，封装了与大模型API交互和视频处理的核心功能"""
    
    def __init__(self, model_api_url: str, api_key: str = None, prompt: str = None, timeout: int = 30, inference_tool: str = "LMStudio", model: str = "qwen/qwen3-vl-8b"):
        """初始化自动标注器
        
        Args:
            model_api_url: 大模型API地址
            api_key: API密钥（如果需要）
            prompt: 自定义提示词
            timeout: HTTP请求超时时间（秒）
            inference_tool: 推理工具，支持LMStudio、vLLM、ollama
            model: 模型名称
        """
        self.model_api_url = model_api_url
        self.api_key = api_key
        self.session = requests.Session()
        self.timeout = timeout
        self.inference_tool = inference_tool
        self.model = model
        # 默认提示词
        self.default_prompt = "检测图中物体，返回JSON：{\"detections\":[{\"label\":\"类别\",\"confidence\":0.9,\"bbox\":[x1,y1,x2,y2]}]}"
        # 使用用户自定义提示词或默认提示词
        self.prompt = prompt if prompt else self.default_prompt
        
        # 定义颜色映射（不同类别使用不同颜色）
        self.colors = {
            "person": (0, 255, 0),
            "car": (255, 0, 0),
            "bicycle": (0, 0, 255),
            "dog": (255, 255, 0),
            "cat": (255, 0, 255),
            "人": (0, 255, 0),
            "车": (255, 0, 0),
            "自行车": (0, 0, 255),
            "狗": (255, 255, 0),
            "猫": (255, 0, 255),
            "default": (0, 255, 255)
        }
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """调用大模型API分析图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            大模型返回的分析结果
        """
        import base64
        import numpy as np
        
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 确保API地址以正确的端点结尾
        api_endpoint = self.model_api_url
        if api_endpoint.endswith("/v1"):
            api_endpoint = f"{api_endpoint}/chat/completions"
        elif not api_endpoint.endswith("/chat/completions"):
            api_endpoint = f"{api_endpoint.rstrip('/')}/v1/chat/completions"
        
        # 读取图像并压缩，减少base64编码后的大小
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 压缩图像（调整大小）
        max_size = 640  # 最大边长
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为JPEG格式，降低质量
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        
        # 构建API请求体
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": self.prompt
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "text"
            }
        }
        
        # 发送请求
        try:
            response = self.session.post(api_endpoint, headers=headers, json=payload, timeout=self.timeout)
            
            # 记录请求详情以便调试
            logging.info(f"发送API请求到: {api_endpoint}")
            logging.info(f"请求头: {headers}")
            logging.info(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
            
            # 检查响应状态码
            if not response.ok:
                # 记录响应详情
                logging.error(f"API请求失败，状态码: {response.status_code}")
                logging.error(f"响应内容: {response.text}")
                raise Exception(f"API请求失败，状态码: {response.status_code}，响应: {response.text[:200]}...")
            
            result = response.json()
            logging.info(f"API响应: {json.dumps(result, ensure_ascii=False)}")
            
            # 解析API返回的结果
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                # 尝试解析JSON内容
                try:
                    # 去除Markdown格式标记
                    if content.startswith('```json'):
                        content = content[7:]  # 移除开头的```json
                    if content.endswith('```'):
                        content = content[:-3]  # 移除结尾的```
                    content = content.strip()  # 去除首尾空白
                    
                    return json.loads(content)
                except json.JSONDecodeError:
                    error_msg = f"无法解析模型返回的JSON: {content}"
                    logging.error(error_msg)
                    raise Exception(error_msg)
            
            return {"detections": []}
        except requests.exceptions.ConnectionError as e:
            error_msg = f"无法连接到API服务器: {str(e)}. 请检查API地址是否正确，服务器是否正在运行。"
            logging.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.Timeout as e:
            error_msg = f"API请求超时: {str(e)}. 请检查网络连接或增加超时时间。"
            logging.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求异常: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"分析图像失败: {str(e)}"
            logging.error(f"分析图像 {image_path} 失败: {e}")
            logging.error(f"使用的API端点: {api_endpoint}")
            raise Exception(error_msg)
    
    def render_detections(self, image_path: str, detections: List[Dict[str, Any]]) -> str:
        """将检测结果渲染到图像上
        
        Args:
            image_path: 原始图像路径
            detections: 检测结果列表
            
        Returns:
            渲染后的图像路径
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 渲染检测框和标签
        for detection in detections:
            # 解析检测结果
            label = detection.get("label", "unknown")
            confidence = detection.get("confidence", 0.0)
            bbox = detection.get("bbox", [0, 0, 0, 0])
            
            # 转换为整数坐标
            x1, y1, x2, y2 = map(int, bbox)
            
            # 获取颜色
            color = self.colors.get(label, self.colors["default"])
            
            # 绘制检测框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签和置信度（支持中文）
            label_text = f"{label}: {confidence:.2f}"
            
            # 尝试使用PIL库渲染中文
            try:
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np
                
                # 转换为PIL图像
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                # 加载默认中文字体或指定字体文件
                try:
                    # 尝试使用系统默认中文字体
                    font = ImageFont.truetype("simhei.ttf", 16)
                except IOError:
                    # 如果没有找到，使用PIL默认字体
                    font = ImageFont.load_default()
                
                # 绘制文本
                text_x = x1
                text_y = y1 - 20 if y1 > 20 else y1 + 20
                draw.text((text_x, text_y), label_text, font=font, fill=tuple(color[::-1]))
                
                # 转换回OpenCV图像
                image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                # 如果PIL渲染失败，使用OpenCV默认渲染（可能会有乱码）
                logging.warning(f"中文渲染失败，使用默认渲染: {e}")
                cv2.putText(image, label_text, (x1, y1 - 10 if y1 > 10 else y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存渲染后的图像
        rendered_path = image_path.replace(".jpg", "_labeled.jpg")
        cv2.imwrite(rendered_path, image)
        return rendered_path
    
    def synthesize_video(self, frames: List[Dict[str, Any]], output_path: str, fps: float) -> bool:
        """合成视频，将选定的帧合成为一个完整的视频
        
        Args:
            frames: 帧数据列表，每个元素包含frame_id、frame和timestamp
            output_path: 输出视频路径
            fps: 视频帧率
            
        Returns:
            视频合成是否成功
        """
        if not frames:
            logging.error("没有帧数据，无法合成视频")
            return False
        
        try:
            # 获取第一帧的尺寸
            first_frame = frames[0]['frame']
            height, width = first_frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logging.error(f"无法打开视频写入器: {output_path}")
                return False
            
            # 写入所有帧
            for frame_data in frames:
                out.write(frame_data['frame'])
            
            # 释放资源
            out.release()
            logging.info(f"✅ 视频合成成功: {output_path}")
            return True
        except Exception as e:
            logging.error(f"合成视频失败: {e}")
            return False
    
    def process_video(self, video_path: str, output_dir: str, frame_interval: int = 1, save_rendered: bool = True):
        """处理视频完整流程，支持本地视频和RTSP流
        
        Args:
            video_path: 视频文件路径或RTSP流地址
            output_dir: 输出目录
            frame_interval: 抽帧间隔
            save_rendered: 是否保存渲染后的帧
        """
        # 记录开始时间
        start_time = datetime.now()
        logging.info(f"🚀 开始处理视频流: {video_path}")
        logging.info(f"📁 输出目录: {output_dir}")
        logging.info(f"⏱️  抽帧间隔: {frame_interval}")
        logging.info(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 创建输出目录
        raw_frames_dir = os.path.join(output_dir, "raw_frames")
        labeled_frames_dir = os.path.join(output_dir, "labeled_frames")
        os.makedirs(raw_frames_dir, exist_ok=True)
        if save_rendered:
            os.makedirs(labeled_frames_dir, exist_ok=True)
        
        logging.info(f"📁 原始帧目录: {raw_frames_dir}")
        if save_rendered:
            logging.info(f"📁 渲染帧目录: {labeled_frames_dir}")
        
        # 初始化变量
        cap = None
        frame_count = 0
        processed_count = 0
        is_rtsp = video_path.lower().startswith("rtsp://")
        max_reconnect_attempts = 50  # 最大重连次数，0表示无限重试
        reconnect_delay = 5  # 重连延迟（秒）
        reconnect_count = 0
        last_status_time = datetime.now()  # 上次输出状态的时间
        status_interval = 60  # 状态输出间隔（秒）
        
        try:
            while True:
                try:
                    # 检查是否需要打开或重新打开视频流
                    if cap is None or not cap.isOpened():
                        if reconnect_count > 0:
                            logging.info(f"🔄 尝试重新连接 RTSP 流... (尝试 {reconnect_count}/{max_reconnect_attempts if max_reconnect_attempts > 0 else '无限'})")
                        else:
                            logging.info(f"📡 打开视频流: {video_path}")
                        
                        # 打开视频或RTSP流
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            raise ValueError(f"无法打开视频流: {video_path}")
                        
                        if reconnect_count > 0:
                            logging.info("✅ RTSP 流重新连接成功")
                            reconnect_count = 0  # 重置重连计数
                    
                    # 读取一帧
                    cap.grab()  # 只抓取帧，不解码，提高响应速度
                    ret, frame = cap.retrieve()  # 解码帧
                    
                    if not ret:
                        if is_rtsp:
                            # RTSP流中断，尝试重连
                            logging.info(f"⚠️  RTSP 流中断，{reconnect_delay}秒后尝试重连...")
                            
                            # 关闭当前视频流
                            if cap is not None:
                                cap.release()
                                cap = None
                            
                            # 增加重连计数
                            reconnect_count += 1
                            
                            # 检查是否达到最大重连次数
                            if max_reconnect_attempts > 0 and reconnect_count > max_reconnect_attempts:
                                logging.error(f"❌ 达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                                break
                            
                            # 等待重连延迟
                            time.sleep(reconnect_delay)
                            continue  # 跳过当前循环，尝试重新连接
                        else:
                            # 本地视频文件结束
                            logging.info("✅ 视频流读取完成")
                            break
                    
                    # 按照指定间隔处理帧
                    if frame_count % frame_interval == 0:
                        logging.info(f"🔄 处理帧 #{frame_count}")
                        
                        # 定义统一的文件名
                        frame_filename = f"frame_{frame_count:06d}.jpg"
                        
                        # 保存临时帧用于处理
                        temp_frame_path = f"temp_{frame_filename}"
                        cv2.imwrite(temp_frame_path, frame)
                        
                        try:
                            # 分析图像（同步处理，阻塞等待结果）
                            result = self.analyze_image(temp_frame_path)
                            
                            # 解析检测结果
                            detections = result.get("detections", [])
                            if isinstance(detections, dict):
                                detections = [detections]
                            
                            # 仅当检测到至少一个目标时，才保存图片
                            if detections and len(detections) > 0:
                                logging.info(f"✅ 检测到 {len(detections)} 个目标")
                                
                                # 保存原始未渲染帧
                                raw_frame_path = os.path.join(raw_frames_dir, frame_filename)
                                cv2.imwrite(raw_frame_path, frame)
                                logging.info(f"✅ 已保存原始帧: {raw_frame_path}")
                                
                                # 保存渲染后的帧
                                if save_rendered:
                                    # 渲染检测结果
                                    rendered_path = self.render_detections(temp_frame_path, detections)
                                    
                                    # 移动渲染后的帧到最终目录，保持与原始帧相同的文件名
                                    final_path = os.path.join(labeled_frames_dir, frame_filename)
                                    os.rename(rendered_path, final_path)
                                    logging.info(f"✅ 已保存标注帧: {final_path}")
                                
                                processed_count += 1
                            else:
                                logging.info(f"ℹ️  未检测到目标，跳过保存")
                        except KeyboardInterrupt:
                            logging.info("\n⚠️  用户中断处理")
                            # 删除未处理完的临时文件
                            if os.path.exists(temp_frame_path):
                                os.remove(temp_frame_path)
                            raise  # 重新抛出异常，让外层处理
                        
                        # 删除临时文件
                        if os.path.exists(temp_frame_path):
                            os.remove(temp_frame_path)
                    
                    frame_count += 1
                    
                    # 定期输出状态信息
                    current_time = datetime.now()
                    if (current_time - last_status_time).total_seconds() >= status_interval:
                        # 计算运行时长
                        elapsed = current_time - start_time
                        # 计算处理速度
                        fps = processed_count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                        
                        logging.info(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"运行时长: {str(elapsed).split('.')[0]} | "
                              f"总帧数: {frame_count} | "
                              f"已处理: {processed_count}帧 | "
                              f"处理速度: {fps:.2f}帧/秒")
                        last_status_time = current_time
                    
                    # 短暂休眠，提高中断响应速度
                    time.sleep(0.001)
                    
                except KeyboardInterrupt:
                    logging.info("\n⚠️  用户中断处理")
                    raise  # 重新抛出异常，让外层处理
                except Exception as e:
                    if is_rtsp:
                        # RTSP流出现异常，尝试重连
                        logging.warning(f"⚠️  RTSP 流异常: {e}，{reconnect_delay}秒后尝试重连...")
                        
                        # 关闭当前视频流
                        if cap is not None:
                            cap.release()
                            cap = None
                        
                        # 增加重连计数
                        reconnect_count += 1
                        
                        # 检查是否达到最大重连次数
                        if max_reconnect_attempts > 0 and reconnect_count > max_reconnect_attempts:
                            logging.error(f"❌ 达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                            raise
                        
                        # 等待重连延迟
                        time.sleep(reconnect_delay)
                        continue  # 跳过当前循环，尝试重新连接
                    else:
                        # 本地视频文件异常，直接抛出
                        raise
        
        except KeyboardInterrupt:
            logging.info("\n🛑 正在停止处理...")
        except Exception as e:
            logging.error(f"❌ 处理异常: {e}")
        finally:
            # 确保视频流被释放
            if cap is not None and cap.isOpened():
                cap.release()
                logging.info("✅ 视频流已释放")
        
        # 计算结束时间和总运行时长
        end_time = datetime.now()
        total_elapsed = end_time - start_time
        
        logging.info(f"\n" + "=" * 60)
        logging.info(f"📊 完整处理统计:")
        logging.info(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"📅 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"⏱️  总运行时长: {str(total_elapsed).split('.')[0]}")
        logging.info(f"📈 总帧数: {frame_count}")
        logging.info(f"✅ 已处理: {processed_count}帧")
        logging.info(f"📊 处理比例: {processed_count / frame_count * 100:.1f}%" if frame_count > 0 else "📊 处理比例: 0%")
        logging.info(f"⚡ 平均速度: {processed_count / total_elapsed.total_seconds():.2f}帧/秒" if total_elapsed.total_seconds() > 0 else "⚡ 平均速度: 0帧/秒")
        logging.info(f"📁 输出目录: {output_dir}")
        logging.info("=" * 60)
        logging.info("✅ 处理已停止")
    
    def process_video_for_segments(self, video_path: str, output_dir: str, frame_interval: int = 1):
        """处理视频并合成检测目标前后1秒的视频片段
        
        Args:
            video_path: 视频文件路径或RTSP流地址
            output_dir: 输出目录
            frame_interval: 抽帧间隔
        """
        # 记录开始时间
        start_time = datetime.now()
        logging.info(f"🚀 开始处理视频流: {video_path}")
        logging.info(f"📁 输出目录: {output_dir}")
        logging.info(f"⏱️  抽帧间隔: {frame_interval}")
        logging.info(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 创建视频输出目录
        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        logging.info(f"📁 视频输出目录: {videos_dir}")
        
        # 初始化变量
        cap = None
        frame_count = 0
        processed_count = 0
        is_rtsp = video_path.lower().startswith("rtsp://")
        max_reconnect_attempts = 50  # 最大重连次数，0表示无限重试
        reconnect_delay = 5  # 重连延迟（秒）
        reconnect_count = 0
        last_status_time = datetime.now()  # 上次输出状态的时间
        status_interval = 60  # 状态输出间隔（秒）
        
        try:
            # 打开视频流获取帧率
            cap_temp = cv2.VideoCapture(video_path)
            if not cap_temp.isOpened():
                logging.error(f"无法打开视频流: {video_path}")
                return
            
            # 获取视频帧率
            fps = cap_temp.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25  # 默认帧率
            cap_temp.release()
            
            logging.info(f"🎬 视频帧率: {fps:.2f} FPS")
            
            # 计算每秒钟的帧数
            frames_per_second = int(round(fps))
            
            # 计算需要保留的帧数：前后1秒，加上当前帧
            frames_to_keep = frames_per_second * 2 + 1
            
            # 创建帧队列，用于保存原始帧
            frame_queue = deque(maxlen=frames_to_keep)
            
            # 重新打开视频流进行处理
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"无法重新打开视频流: {video_path}")
                return
            
            while True:
                try:
                    # 检查视频流是否打开
                    if not cap.isOpened():
                        if is_rtsp:
                            logging.info(f"⚠️  RTSP 流中断，{reconnect_delay}秒后尝试重连...")
                            
                            # 增加重连计数
                            reconnect_count += 1
                            
                            # 检查是否达到最大重连次数
                            if max_reconnect_attempts > 0 and reconnect_count > max_reconnect_attempts:
                                logging.error(f"❌ 达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                                break
                            
                            # 等待重连延迟
                            time.sleep(reconnect_delay)
                            
                            # 重新打开视频流
                            cap.release()
                            cap = cv2.VideoCapture(video_path)
                            if cap.isOpened():
                                logging.info("✅ RTSP 流重新连接成功")
                                reconnect_count = 0  # 重置重连计数
                                # 清空帧队列
                                frame_queue.clear()
                                # 重新获取帧率
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                if fps <= 0:
                                    fps = 25  # 默认帧率
                                frames_per_second = int(round(fps))
                                frames_to_keep = frames_per_second * 2 + 1
                                frame_queue = deque(maxlen=frames_to_keep)
                            continue
                        else:
                            # 本地视频文件结束
                            logging.info("✅ 视频流读取完成")
                            break
                    
                    # 读取一帧
                    ret, frame = cap.read()
                    
                    if not ret:
                        if is_rtsp:
                            logging.info(f"⚠️  RTSP 流中断，{reconnect_delay}秒后尝试重连...")
                            
                            # 关闭当前视频流
                            cap.release()
                            cap = None
                            
                            # 增加重连计数
                            reconnect_count += 1
                            
                            # 检查是否达到最大重连次数
                            if max_reconnect_attempts > 0 and reconnect_count > max_reconnect_attempts:
                                logging.error(f"❌ 达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                                break
                            
                            # 等待重连延迟
                            time.sleep(reconnect_delay)
                            continue
                        else:
                            # 本地视频文件结束
                            logging.info("✅ 视频流读取完成")
                            break
                    
                    # 保存原始帧到队列
                    frame_data = {
                        'frame_id': frame_count,
                        'frame': frame.copy(),
                        'timestamp': time.time()
                    }
                    frame_queue.append(frame_data)
                    
                    # 按照指定间隔处理帧
                    if frame_count % frame_interval == 0:
                        logging.info(f"🔄 处理帧 #{frame_count}")
                        
                        # 保存临时帧用于分析
                        temp_frame_path = f"temp_frame_{frame_count:06d}.jpg"
                        cv2.imwrite(temp_frame_path, frame)
                        
                        try:
                            # 分析图像（同步处理，阻塞等待结果）
                            result = self.analyze_image(temp_frame_path)
                            
                            # 解析检测结果
                            detections = result.get("detections", [])
                            if isinstance(detections, dict):
                                detections = [detections]
                            
                            # 如果检测到目标
                            if detections and len(detections) > 0:
                                logging.info(f"✅ 检测到 {len(detections)} 个目标")
                                
                                # 计算需要提取的帧范围：当前帧前后1秒
                                target_frame_id = frame_count
                                start_frame_id = max(0, target_frame_id - frames_per_second)
                                end_frame_id = target_frame_id + frames_per_second
                                
                                # 从队列中获取符合条件的帧
                                selected_frames = []
                                for frame_data in frame_queue:
                                    if start_frame_id <= frame_data['frame_id'] <= end_frame_id:
                                        selected_frames.append(frame_data)
                                
                                # 按照frame_id排序
                                selected_frames.sort(key=lambda x: x['frame_id'])
                                
                                # 生成输出视频文件名
                                video_filename = f"video_{frame_count:06d}.mp4"
                                video_output_path = os.path.join(videos_dir, video_filename)
                                
                                # 合成视频
                                self.synthesize_video(selected_frames, video_output_path, fps)
                                processed_count += 1
                            else:
                                logging.info(f"ℹ️  未检测到目标，跳过视频合成")
                        except KeyboardInterrupt:
                            logging.info("\n⚠️  用户中断处理")
                            # 删除未处理完的临时文件
                            if os.path.exists(temp_frame_path):
                                os.remove(temp_frame_path)
                            raise  # 重新抛出异常，让外层处理
                        
                        # 删除临时文件
                        if os.path.exists(temp_frame_path):
                            os.remove(temp_frame_path)
                    
                    frame_count += 1
                    
                    # 定期输出状态信息
                    current_time = datetime.now()
                    if (current_time - last_status_time).total_seconds() >= status_interval:
                        # 计算运行时长
                        elapsed = current_time - start_time
                        # 计算处理速度
                        fps_processed = processed_count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                        
                        logging.info(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"运行时长: {str(elapsed).split('.')[0]} | "
                              f"总帧数: {frame_count} | "
                              f"已合成: {processed_count}个视频 | "
                              f"处理速度: {fps_processed:.2f}个视频/秒")
                        last_status_time = current_time
                    
                    # 短暂休眠，提高中断响应速度
                    time.sleep(0.001)
                    
                except KeyboardInterrupt:
                    logging.info("\n⚠️  用户中断处理")
                    raise  # 重新抛出异常，让外层处理
                except Exception as e:
                    if is_rtsp:
                        # RTSP流出现异常，尝试重连
                        logging.warning(f"⚠️  RTSP 流异常: {e}，{reconnect_delay}秒后尝试重连...")
                        
                        # 关闭当前视频流
                        cap.release()
                        cap = None
                        
                        # 增加重连计数
                        reconnect_count += 1
                        
                        # 检查是否达到最大重连次数
                        if max_reconnect_attempts > 0 and reconnect_count > max_reconnect_attempts:
                            logging.error(f"❌ 达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                            break
                        
                        # 等待重连延迟
                        time.sleep(reconnect_delay)
                        continue  # 跳过当前循环，尝试重新连接
                    else:
                        # 本地视频文件异常，直接抛出
                        raise
        
        except KeyboardInterrupt:
            logging.info("\n🛑 正在停止处理...")
        except Exception as e:
            logging.error(f"❌ 处理异常: {e}")
        finally:
            # 确保视频流被释放
            if cap is not None and cap.isOpened():
                cap.release()
                logging.info("✅ 视频流已释放")
        
        # 计算结束时间和总运行时长
        end_time = datetime.now()
        total_elapsed = end_time - start_time
        
        logging.info(f"\n" + "=" * 60)
        logging.info(f"📊 完整处理统计:")
        logging.info(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"📅 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"⏱️  总运行时长: {str(total_elapsed).split('.')[0]}")
        logging.info(f"📈 总帧数: {frame_count}")
        logging.info(f"✅ 已合成视频: {processed_count}个")
        logging.info(f"📊 处理比例: {processed_count / frame_count * 100:.1f}%" if frame_count > 0 else "📊 处理比例: 0%")
        logging.info(f"⚡ 平均速度: {processed_count / total_elapsed.total_seconds():.2f}个视频/秒" if total_elapsed.total_seconds() > 0 else "⚡ 平均速度: 0个视频/秒")
        logging.info(f"📁 视频输出目录: {videos_dir}")
        logging.info("=" * 60)
        logging.info("✅ 处理已停止")
