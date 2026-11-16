# ytlabel（关键点标注分支）

**语言 / Language：** [简体中文](README.md) | [English](README_en.md)

> **本仓库是上游 [yuturuishi/ytlabel](https://github.com/yuturuishi/ytlabel) 的个人分支**，基于 v4.0 增加了**关键点标注**、**数据集目录切换**、**YOLO pose 格式导出**。上游原有功能、界面与 MIT 协议保持不变，版权归原作者所有。

开源（MIT）图像标注工具：**标准标注 + 关键点标注 + 数据集导出**。基于 Python + Flask，跨平台支持 Windows / Linux / Mac。模型训练不在本工具内，请使用专用训练工具 [ytstudio](https://github.com/yuturuishi/ytstudio)。

---

## 本分支新增（相对上游 v4.0）

| 功能 | 说明 | 入口 |
|---|---|---|
| **关键点标注** | 矩形框 + 关键点（bbox + kpts），可设每框固定点数；画完框自动进入点模式，点满自动回矩形工具 | 右侧「关键点」面板 / 快捷键 `K` |
| **数据集目录切换** | 文件管理页可浏览任意目录（盘符下拉、上级目录），一键「设为数据集」；标注跟着数据集走，存在 `<目录>/.ytlabel/` | 顶部「文件管理」 |
| **启动指定目录** | `--data-dir` 参数或 `YTLABEL_DATA_DIR` 环境变量；选择会被记住，下次启动自动恢复 | 命令行 |
| **YOLO pose 导出** | 只要有任何关键点，导出即自动切 pose 格式（`data.yaml` 带 `kpt_shape: [N, 3]`，标签为 `cls cx cy w h x y v ...`） | 「导出数据集」 |
| **绝对路径上传** | 切到外部目录后，文件上传 / 新建文件夹也支持绝对路径 | 「文件管理」 |

### 关键点标注怎么用

1. 画一个矩形框，**点一下把它选中**（框变红）
2. 右侧「关键点」面板里填点名，逗号分隔，例如 `center,stem`
3. 「每个框点数」填一个数字（比如 `2`；填 `0` 表示不限、手动控制）
4. 画完框会**自动进入关键点模式**，在图上依次点位置
5. 点满后**自动退回矩形工具**，直接画下一个框；想手动改某个框，选中它按 `K` 即可

导出后标签形如：

```text
# cls cx cy w h  kx1 ky1 v1  kx2 ky2 v2
0 0.312500 0.520833 0.312500 0.625000  0.312500 0.520833 2  0.312500 0.833333 2
```

`v=2` 表示该点已标，`0 0 0` 表示这个框缺少该关键点（自动补齐）。

### 训练 pose 模型

```bash
yolo pose train model=yolo11n-pose.pt data=<导出目录>/data.yaml epochs=100 imgsz=640 batch=4
```

> 注意：keypoint 顺序即语义（第 1 个点永远是同一个部位），标注中途不要改变点数或顺序。

---

## 快速开始

环境要求：Python 3.9+（实测 3.10）、现代浏览器。

```bash
# 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / Mac

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动服务（默认数据集目录：uploads/samples）
python app.py --host 0.0.0.0 --port 9924

# 或者直接指定数据集目录（标注会存到该目录下的 .ytlabel/）
python app.py --host 0.0.0.0 --port 9924 --data-dir "D:/data/my_dataset"
```

浏览器访问 http://127.0.0.1:9924 即可。

---

## 功能（上游原有）

- **标注**：矩形、多边形等多种类型；支持图片、视频、LabelMe 数据集导入
- **AI 自动标注**：对接你自行配置的大模型 API（LMStudio、vLLM、ollama、阿里云等）
- **数据集导出**：YOLO 格式，可自定义训练 / 验证 / 测试比例
- **文件管理**：内置文件浏览、上传、下载
- **离线部署**：全部静态资源本地化，支持内网离线运行
- **模型训练**：不在本工具内，请使用 [ytstudio](https://github.com/yuturuishi/ytstudio)

**快捷键**：`Ctrl+S` 保存标注 ｜ `Ctrl+Shift+D` 清除标注 ｜ `K` 切换关键点模式

---

## 软件截图

<img width="720" alt="1" src="https://raw.giteeusercontent.com/yuturuishi/images/raw/master/xclabel/v3.0/1.png">
<img width="720" alt="2" src="https://raw.giteeusercontent.com/yuturuishi/images/raw/master/xclabel/v3.0/2.png">
<img width="720" alt="3" src="https://raw.giteeusercontent.com/yuturuishi/images/raw/master/xclabel/v3.0/3.png">

---

## 项目结构

```
ytlabel/
├── app.py                 # 主应用（含数据集目录切换、pose 导出）
├── AiUtils.py             # AI 自动标注工具类
├── requirements.txt       # 依赖列表
├── ytlabel.spec           # PyInstaller 打包配置
├── logo.ico               # 应用图标
├── static/                # 静态资源（样式、脚本、Socket.IO 本地库）
├── templates/             # 页面模板：index.html / ai_config.html / file_manager.html
└── uploads/               # 默认数据目录（运行时自动创建）
    ├── annotations/       # 默认数据集的标注
    ├── samples/           # 默认数据集的图片
    └── active_dataset.json# 记住上次切换的数据集目录
```

切换到外部数据集目录时，该目录下会生成 `.ytlabel/`（`annotations.json` + `classes.json`），与默认数据集互不干扰。

---

## 技术栈

Flask + Flask-SocketIO ｜ HTML/CSS/JS ｜ OpenCV/PIL ｜ Socket.IO

---

## 版本历史

### 本分支 v4.1-keypoint — 2026/11/16
- 新增关键点标注：每框点数设定、画框自动进入、点满自动退回矩形工具、`K` 快捷键
- 新增数据集目录切换：文件管理页盘符 / 上级目录 / 设为数据集，`--data-dir` 与 `YTLABEL_DATA_DIR`
- 导出支持 YOLO pose 格式（`kpt_shape`、关键点 `x y v`，缺失自动补 0）
- 上传 / 新建文件夹支持绝对路径

### v4.0 — 2026/10/04（上游）
- **移除内置模型训练**：专注「数据标注 + 数据集导出」，训练改用 [ytstudio](https://github.com/yuturuishi/ytstudio)
- **彻底移除 Ultralytics 依赖**：删除 `pre_models/` 目录与全部官方 YOLO 预训练权重，仓库零 ultralytics 残留
- **项目更名**：xclabel → ytlabel（含 `ytlabel.spec`、页面标题、本地设置项）
- **批量删除优化**：消除标注表 O(N²) 反复读写，新增全局加载遮罩
- **文档整合**：CHANGELOG 并入 README；设置面板精简；合规说明更新

### v3.0 — 2026/05/19（上游）
- 新增独立训练面板：YOLO11 训练全流程、断点续训、模型测试与下载
- 训练状态持久化：页面刷新 / SocketIO 重连后自动恢复进度
- 预训练模型目录迁移至 `pre_models`；Socket.IO 本地化，支持离线部署

### v2.x（上游）
- **v2.7** — 2026/01/05：标注文件迁移至 `uploads/annotations/`，API 配置迁移至 `uploads/config/`
- **v2.6** — 2026/01/04：新增 AI 自动标注（大模型 / 阿里云）
- **v2.4** — 2026/01/03：优化 RTSP 流与 SocketIO 断连处理；修复打包运行报错
- **v2.3** — 2025/12/30：新增文件管理系统（导航、预览、批量下载/删除、上传）
- **v2.2** — 2025/12/30：标注完成后原图 / 渲染图对比，支持点击放大
- **v2.1** — 2025/12/01：新增 YOLO11 集成；优化标注工具、标签管理与数据集导出
- **v2.0** — 2025/11/29：从 Django 迁移到 Flask，支持视频帧提取、LabelMe 导入、YOLO 导出

### v1.0 — 2025/02/17（上游）
- 基础图像标注（矩形、多边形），Python + Django
- v1.0 分支：https://gitee.com/yuturuishi/ytlabel/tree/v1.0

---

## 授权协议

本项目自有代码使用 **MIT** 协议（见 [LICENSE](LICENSE)），版权归原作者 **yuturuishi（宇图瑞视）** 所有；本分支的修改同样以 MIT 发布。第三方库遵循其各自协议。

- **AI 自动标注**：通过你自行配置的大模型 API（LMStudio、vLLM、ollama、阿里云等本地 / 私有服务）完成，**不依赖 Ultralytics 库**。
- **网络请求**：仅与本机服务及你配置的本地 / 私有大模型 API 通信，**不向任何外部服务器回传数据、标注内容或运行信息**。
- **数据**：本仓库不包含任何标注数据与图片，标注数据保存在本地 `uploads/` 或你指定的数据集目录中。
- **免责**：部署环境、所用模型与数据的合法性由使用者自行负责。ytlabel 按「现状」提供，不作任何担保。
