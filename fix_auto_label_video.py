# 修复后的auto_label_video函数
@app.route('/api/auto-label/video', methods=['POST'])
def auto_label_video():
    """视频自动标注"""
    try:
        import os
        import time
        from collections import deque
        import logging
        
        # 获取请求数据
        data = request.json
        video_path = data.get('video_path')
        frame_interval = int(data.get('frame_interval', 10))
        output_dir = data.get('output_dir', 'output')
        api_config = data.get('api_config', {})
        
        if not video_path:
            return jsonify({'success': False, 'error': 'No video path provided'}), 400
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        raw_dir = os.path.join(output_dir, 'raw_frames')
        labeled_dir = os.path.join(output_dir, 'labeled_frames')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(labeled_dir, exist_ok=True)
        
        # 获取API配置
        api_url = api_config.get('apiUrl', 'http://192.168.1.105:1234/v1')
        api_key = api_config.get('apiKey', '')
        timeout = int(api_config.get('timeout', 30))
        prompt = api_config.get('prompt', '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}')
        model = api_config.get('model', 'qwen/qwen3-vl-8b')
        inference_tool = api_config.get('inferenceTool', 'LMStudio')
        
        # 初始化AIAutoLabeler
        labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)
        
        # 打开视频流
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': f'Failed to open video: {video_path}'}), 400
        
        frame_count = 0
        processed_count = 0
        total_detections = 0
        
        # 处理视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按照指定间隔处理帧
            if frame_count % frame_interval == 0:
                # 保存原始帧
                frame_filename = f"frame_{frame_count:06d}.jpg"
                raw_frame_path = os.path.join(raw_dir, frame_filename)
                cv2.imwrite(raw_frame_path, frame)
                
                # 调用API进行标注
                try:
                    result = labeler.analyze_image(raw_frame_path)
                    detections = result.get("detections", [])
                    if isinstance(detections, dict):
                        detections = [detections]
                except Exception as e:
                    error_msg = f"处理视频帧失败: {str(e)}"
                    logging.error(error_msg)
                    return jsonify({
                        'success': False,
                        'error': error_msg,
                        'processed': processed_count,
                        'detections': total_detections,
                        'output_dir': output_dir
                    }), 500
                
                # 渲染检测结果
                rendered_path = labeler.render_detections(raw_frame_path, detections)
                
                # 移动渲染后的图片到输出目录
                labeled_path = os.path.join(labeled_dir, frame_filename)
                # 如果目标文件已存在，先删除
                if os.path.exists(labeled_path):
                    os.remove(labeled_path)
                os.rename(rendered_path, labeled_path)
                
                processed_count += 1
                total_detections += len(detections)
            
            frame_count += 1
        
        # 释放资源
        cap.release()
        
        return jsonify({
            'success': True,
            'processed': processed_count,
            'detections': total_detections,
            'output_dir': output_dir
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500