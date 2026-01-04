# 修复后的auto_label_image函数
@app.route('/api/auto-label/image', methods=['POST'])
def auto_label_image():
    """图片自动标注"""
    try:
        import os
        import tempfile
        import logging
        
        # 获取表单数据
        files = request.files.getlist('images')
        output_dir = request.form.get('output_dir', 'output')
        api_url = request.form.get('api_url', 'http://192.168.1.105:1234/v1')
        api_key = request.form.get('api_key', '')
        timeout = int(request.form.get('timeout', 30))
        prompt = request.form.get('prompt', '检测图中物体，返回JSON：{"detections":[{"label":"类别","confidence":0.9,"bbox":[x1,y1,x2,y2]}]}')
        inference_tool = request.form.get('inferenceTool', 'LMStudio')
        
        if not files:
            return jsonify({'success': False, 'error': 'No image files provided'}), 400
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        raw_dir = os.path.join(output_dir, 'raw_frames')
        labeled_dir = os.path.join(output_dir, 'labeled_frames')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(labeled_dir, exist_ok=True)
        
        processed_count = 0
        total_detections = 0
        
        # 初始化图片列表，用于存储每张图片的处理结果和Base64数据
        images = []
        
        # 获取模型配置
        model = request.form.get('model', 'qwen/qwen3-vl-8b')
        
        # 初始化AIAutoLabeler
        labeler = AIAutoLabeler(api_url, api_key, prompt, timeout, inference_tool, model)
        
        # 处理每张图片
        for file in files:
            if file.filename == '':
                continue
            
            # 保存原始图片
            filename = os.path.basename(file.filename)
            raw_path = os.path.join(raw_dir, filename)
            file.save(raw_path)
            
            # 调用API进行标注
            try:
                result = labeler.analyze_image(raw_path)
                detections = result.get("detections", [])
                if isinstance(detections, dict):
                    detections = [detections]
            except Exception as e:
                error_msg = f"处理图片失败: {str(e)}"
                logging.error(error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'processed': processed_count,
                    'detections': total_detections,
                    'output_dir': output_dir
                }), 500
            
            # 渲染检测结果
            rendered_path = labeler.render_detections(raw_path, detections)
            
            # 移动渲染后的图片到输出目录
            labeled_path = os.path.join(labeled_dir, filename)
            # 如果目标文件已存在，先删除
            if os.path.exists(labeled_path):
                os.remove(labeled_path)
            os.rename(rendered_path, labeled_path)
            
            # 生成原始图片的Base64数据
            import base64
            with open(raw_path, "rb") as f:
                raw_image_data = f.read()
            raw_image_base64 = base64.b64encode(raw_image_data).decode("utf-8")
            raw_image_base64 = f"data:image/jpeg;base64,{raw_image_base64}"
            
            # 生成渲染后图片的Base64数据
            with open(labeled_path, "rb") as f:
                labeled_image_data = f.read()
            labeled_image_base64 = base64.b64encode(labeled_image_data).decode("utf-8")
            labeled_image_base64 = f"data:image/jpeg;base64,{labeled_image_base64}"
            
            # 将图片信息添加到列表
            images.append({
                'filename': filename,
                'original_image': raw_image_base64,
                'labeled_image': labeled_image_base64,
                'detections': len(detections)
            })
            
            processed_count += 1
            total_detections += len(detections)
        
        return jsonify({
            'success': True,
            'processed': processed_count,
            'detections': total_detections,
            'output_dir': output_dir,
            'images': images
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500