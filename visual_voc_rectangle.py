import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os

# --- 配置 ---
# 假设你的文件结构如下：
# ├── my_script.py
# ├── JPEGImages/
# │   └── 000001.jpg
# └── Annotations/
#     └── 000001.xml

IMAGE_DIR = os.path.expanduser("~/Documents/datasets/coco2voc/JPEGImages")  # 你的图片文件夹路径
ANNOTATION_DIR = os.path.expanduser("~/Documents/datasets/coco2voc/Annotations") # 你的标注文件夹路径
IMAGE_NAME = "000000000139" # 你想可视化的图片文件基名

def parse_voc_annotation(annotation_path):
    """
    解析 PASCAL VOC 格式的 XML 标注文件。
    返回图片尺寸和所有对象的边界框信息。
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    # 1. 提取图片尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    boxes = []
    
    # 2. 遍历所有对象
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        # 3. 提取边界框的像素坐标
        # VOC 格式的坐标是左上角和右下角，且通常是从 1 开始计数
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)
        
        # 标注矫正：将 VOC 的 1-based 坐标转换为 0-based
        # 虽然许多库会自动处理，但为安全起见，这里可以减 1（如果需要）
        # x_min = max(0, x_min - 1)
        # y_min = max(0, y_min - 1)
        
        boxes.append({
            'name': class_name,
            'bbox': (x_min, y_min, x_max, y_max)
        })
        
    return img_width, img_height, boxes

def draw_voc_boxes(image_dir, annotation_dir, image_base_name):
    """
    加载图片和标注，并绘制边界框。
    """
    image_path = os.path.join(image_dir, f"{image_base_name}.jpg")
    annotation_path = os.path.join(annotation_dir, f"{image_base_name}.xml")
    
    if not os.path.exists(image_path):
        print(f"错误: 图片文件未找到: {image_path}")
        return
    if not os.path.exists(annotation_path):
        print(f"错误: XML 标注文件未找到: {annotation_path}")
        return

    # 1. 解析标注
    _, _, boxes = parse_voc_annotation(annotation_path)

    # 2. 读取图片
    image = cv2.imread(image_path)
    
    # 3. 绘制每个边界框
    for obj in boxes:
        class_name = obj['name']
        x_min, y_min, x_max, y_max = obj['bbox']
        
        # 绘制颜色（为了演示，我们使用固定颜色）
        color = (255, 0, 0) # 蓝色 BGR 格式
        thickness = 2
        
        # 绘制矩形框 (cv2.rectangle 接收左上角和右下角坐标)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # 绘制类别标签
        label_text = class_name
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # 绘制背景矩形（使文本清晰）
        cv2.rectangle(image, (x_min, y_min - text_size[1] - 5), (x_min + text_size[0], y_min), color, -1)
        
        # 绘制文本
        cv2.putText(image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # 4. 保存结果
    output_path = f"{image_base_name}_voc_boxes.jpg"
    cv2.imwrite(output_path, image)
    print(f"绘图完成，结果已保存到: {output_path}")

# 执行函数
draw_voc_boxes(IMAGE_DIR, ANNOTATION_DIR, IMAGE_NAME)
