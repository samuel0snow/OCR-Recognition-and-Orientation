import cv2
import numpy as np
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz
from PIL import Image

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

if __name__ == "__main__":
    image_path = input("请输入图片路径（例如 photo.jpg）：")
    query = input("请输入要查找的文字片段：")

    print("🔍 正在识别图片文字，请稍候...")
    result = detect_text(image_path)

    matches = search_text(result, query)
    if not matches:
        print("未找到匹配的文字。")
    else:
        print(f"✅ 找到 {len(matches)} 处匹配结果。")
        output = draw_boxes(image_path, matches)
        print(f"结果已保存至：{output}")
