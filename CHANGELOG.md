# 更新日志 / Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号沿用上游命名习惯。

本仓库是上游 [yuturuishi/ytlabel](https://github.com/yuturuishi/ytlabel) 的分支，`4.0` 及更早版本的记录见上游仓库。

## [Unreleased]

### 计划中
- 关键点支持 `flip_idx`，打开左右翻转增强
- 导出时可自定义关键点名称列表（写进 `data.yaml`）
- 关键点模式支持 `Shift+点击` 删除最近一个点

## [4.1-keypoint] - 2026-11-16

### 新增
- **关键点标注**：矩形框 + 关键点（bbox + kpts），一个框可标任意数量关键点
- **每框固定点数**：在「关键点」面板设定数量，画完框自动进入点模式，点满自动退回矩形工具，支持连续作业
- **快捷键 `K`** 切换关键点模式；面板可单独删除某个关键点
- **数据集目录切换**：文件管理页支持盘符下拉、上级目录，一键「设为数据集」；标注存到该目录下的 `.ytlabel/`
- **启动参数**：`--data-dir` / 环境变量 `YTLABEL_DATA_DIR` 指定数据集目录；选择会被记住（`uploads/active_dataset.json`）
- **YOLO pose 导出**：数据里存在关键点时自动导出 pose 格式，`data.yaml` 带 `kpt_shape: [N, 3]`，标签为 `cls cx cy w h x y v ...`，缺失关键点补 `0 0 0`

### 修复
- 「图片文件夹」上传缺少 `webkitdirectory`，只能多选文件、选不了文件夹；现补「选择文件夹」与「选择图片文件」两个入口
- 上传体积超限时服务端返回 HTML 错误页，前端只能显示「上传失败」；现返回 JSON 提示（上限可调，默认 2 GB）
- 前端上传改为分批（每批 100 张 / 150 MB），避免单次请求过大被拒
- 切换到外部目录后，上传文件 / 新建文件夹因只接受相对路径而失败

### 说明
- **关键点顺序即语义**：第 1 个点永远是同一个部位，标注中途不要改变顺序或点数，否则标签错位
- 本仓库不包含图片与标注数据，数据保存在本地 `uploads/` 或你指定的数据集目录

## [4.0] - 2026-10-04（上游）

### 变更
- 移除内置模型训练，专注「数据标注 + 数据集导出」，训练改用 [ytstudio](https://github.com/yuturuishi/ytstudio)
- 彻底移除 Ultralytics 依赖（删除 `pre_models/` 与全部官方 YOLO 预训练权重）
- 项目更名：xclabel → ytlabel（含 `ytlabel.spec`、页面标题、本地设置项）
- 批量删除优化：消除标注表 O(N²) 反复读写，新增全局加载遮罩
- 文档整合：CHANGELOG 并入 README

更早版本（v3.0 / v2.x / v1.0）请见上游仓库的版本历史。
