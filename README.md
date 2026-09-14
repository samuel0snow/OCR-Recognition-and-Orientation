# OCR Recognition and Orientation

离线中文 OCR 关键词定位工具：识别图片中的文字，找到关键词后在原图上用红框紧贴标记该关键词。仓库内已包含检测、识别和方向分类模型，运行时不会下载模型。

## 安装

推荐 Python 3.10，并将依赖安装到项目虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 可视化软件

激活虚拟环境并启动桌面界面：

```powershell
.\.venv\Scripts\Activate.ps1
python ocr_locator_app.py
```

Windows 下也可以直接双击 `启动软件.bat`，它会使用项目的 `.venv` 启动界面。

软件包含两个页面：

- **单张图片**：将图片拖入左侧区域（也可点击选择），输入关键词并开始识别，右侧会预览只框住关键词的结果；图片同时保存至项目的 `outputs/`。
- **批量处理**：选择图片文件夹并输入关键词。程序处理该文件夹第一层中的全部支持图片，并在该文件夹下创建 `outputs/` 保存结果。没有找到关键词的图片也会输出原图副本，处理记录会明确说明。

识别在后台运行，界面不会因 OCR 阻塞；同一次启动会复用 OCR 模型，加快后续图片处理。支持 JPG、JPEG、PNG、BMP 和 WEBP。

## 命令行使用

在已激活的虚拟环境中运行：

```powershell
python ocr_cli.py
```

第二条命令默认处理 `target_img/识别并标记出“对象”.jpg`（也兼容旧的 `examples/` 目录），查找“对象”，并生成 `outputs/对象_标记结果.jpg`。

也可以显式指定图片、关键词与输出位置：

```powershell
python ocr_cli.py "target_img/识别并标记出“对象”.jpg" "对象" -o "outputs/marked.jpg" --report "outputs/report.json"
```

## 行为与鲁棒性

- 使用仓库中的本地模型，支持图片路径和输出路径含中文。
- 自动进行文本方向分类，适合旋转的拍摄图片。
- 默认先做精确子串匹配；若 OCR 有单字误识别，再以 `--threshold`（默认 85）进行模糊匹配。
- 会检查图片、模型、关键词和输出格式，并以非零退出码报告未找到或输入错误。
- 对精确匹配，按关键词在 OCR 文本行中的字符位置计算局部四点框，因此不会覆盖整行；该框会保留原图片的倾斜透视。模糊匹配无法确定单字位置时，才会退回到整行框。

## 输出

`--report` 可额外生成 UTF-8 JSON，记录每处匹配的原始文本、OCR 置信度、匹配分数与四点坐标，便于后续集成。

## 项目结构

```text
ocr_core.py         OCR 模型、识别、匹配和图片标注核心
ocr_cli.py          命令行入口
ocr_locator_app.py  可视化桌面软件
启动软件.bat         Windows 双击启动入口
models/             本地 OCR 模型
target_img/         示例及待处理图片
outputs/            生成结果（不提交到 Git）
```
