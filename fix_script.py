import re

# 读取修复后的函数内容
with open('fix_auto_label.py', 'r', encoding='utf-8') as f:
    new_func = f.read()

# 读取app.py文件内容
with open('app.py', 'r', encoding='utf-8') as f:
    app_content = f.read()

# 定义要替换的函数模式
start_pattern = r'@app.route\(\'/api/auto-label/image\', methods=\[\'POST\'\]\)'
end_pattern = r'@app.route\(\'/api/auto-label/video\', methods=\[\'POST\'\]\)'

# 找到函数的开始和结束位置
start_match = re.search(start_pattern, app_content)
end_match = re.search(end_pattern, app_content)

if start_match and end_match:
    start_pos = start_match.start()
    end_pos = end_match.start()
    
    # 构建新的文件内容
    new_content = app_content[:start_pos] + new_func + '\n\n' + app_content[end_pos:]
    
    # 写入修复后的内容
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print('Auto label function fixed successfully!')
else:
    print('Could not find the auto_label_image function in app.py')
