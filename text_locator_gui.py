import cv2
import numpy as np
import os
import sys
from tkinter import Tk, filedialog, simpledialog, messagebox
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz

# 防止中文输出乱码，兼容打包后的 .exe
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding='utf-8')


def detect_text(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(image_path, cls=True)
    return result[0]

def search_text(result, query, threshold=80):
    matches = []
    for line in result:
        text = line[1][0]
        if fuzz.partial_ratio(query, text) >= threshold:
            matches.append(line)
    return matches

def draw_boxes(image_path, matches, output_path="output.jpg"):
    img = cv2.imread(image_path)
    for line in matches:
        box = np.array(line[0], dtype=np.int32)
        cv2.polylines(img, [box], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(output_path, img)
    return output_path

def main():
    root = Tk()
    root.withdraw()  # 不显示主窗口

    messagebox.showinfo("中文图片文字定位器", "请选择要识别的图片文件。")
    file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("图像文件", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        messagebox.showwarning("取消", "未选择图片。")
        return

    query = simpledialog.askstring("输入文字", "请输入要查找的文字片段：")
    if not query:
        messagebox.showwarning("取消", "未输入文字。")
        return

    messagebox.showinfo("识别中", "正在识别，请稍候...")
    try:
        result = detect_text(file_path)
        matches = search_text(result, query)

        if not matches:
            messagebox.showinfo("结果", "未找到匹配的文字。")
            return

        output_path = os.path.join(os.path.dirname(file_path), "output.jpg")
        draw_boxes(file_path, matches, output_path)
        os.startfile(output_path)

        messagebox.showinfo("完成", f"✅ 找到 {len(matches)} 处匹配结果。\n结果已保存至：\n{output_path}")

    except Exception as e:
        messagebox.showerror("错误", f"程序出错：{e}")

if __name__ == "__main__":
    main()
