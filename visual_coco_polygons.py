import cv2
import os
import numpy as np
import re

"""
This file is used to visualize the polygon ground truth box of the coco dataset.
"""

# 1. 定义文件路径
image_base_name = "000000000139"
image_path = f"/home/lxx/Documents/datasets/coco/images/val2017/{image_base_name}.jpg"
label_path = f"/home/lxx/Documents/datasets/coco/labels/val2017/{image_base_name}.txt"

# 假设的类别名称列表（用于显示文本标签）
# 确保这里的索引对应您的 Class_ID
CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

def load_and_clean_label_content(path):
    """
    从文件读取内容并清理非标注数据（如 ）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"错误: 标注文件 '{path}' 不存在。")
    
    with open(path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式移除 这样的标记
    content_clean = re.sub(r'\\', '', content).strip()
    return content_clean

def draw_polygon_annotations(image_path, label_path, class_names):
    """
    从文件读取图片和多边形标注，并将其绘制在图片上。
    """
    try:
        label_content = load_and_clean_label_content(label_path)
    except FileNotFoundError as e:
        print(e)
        return

    # 1. 读取图片
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图片: {image_path}")
    except Exception as e:
        print(e)
        return

    H, W, _ = image.shape
    print(f"图片尺寸: 宽度={W}, 高度={H}")

    lines = label_content.split('\n')

    for line in lines:
        if not line.strip():
            continue
            
        # 2. 解析和反归一化
        try:
            parts = [float(p) for p in line.split()]
            class_id = int(parts[0])
            # 后续所有数据都是归一化坐标对 (x1, y1, x2, y2, ...)
            coords_flat = parts[1:]
            
        except (ValueError, IndexError):
            print(f"警告: 跳过格式错误的行: {line}")
            continue

        # 检查坐标点是否为偶数个 (x, y)
        if len(coords_flat) % 2 != 0 or len(coords_flat) < 2:
            print(f"警告: 标注行坐标点数量异常，跳过: {line}")
            continue

        # 将一维列表转换为二维数组 [(x1, y1), (x2, y2), ...]
        coords = np.array(coords_flat).reshape(-1, 2)
        
        # 将归一化坐标转换为像素坐标
        # x 坐标乘以 W，y 坐标乘以 H
        coords_pixel = np.array([(int(x * W), int(y * H)) for x, y in coords], dtype=np.int32)

        # 3. 绘制多边形和标签
        # 使用不同的颜色来区分不同类别 (这里只是简单示例)
        color = (0, 255, 0) # 绿色 BGR 格式
        
        # 绘制多边形 (参数 True 表示闭合多边形)
        cv2.polylines(image, [coords_pixel], isClosed=True, color=color, thickness=2)

        # 找到多边形的最小边界框 (用于放置标签)
        x_min = coords_pixel[:, 0].min()
        y_min = coords_pixel[:, 1].min()

        # 添加类别标签
        label_text = class_names.get(class_id, f"Class {class_id}")
        
        # 绘制文本 (放置在多边形左上角)
        cv2.putText(image, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    # 保存结果
    output_path = f"{image_base_name}_with_polygons.jpg"
    cv2.imwrite(output_path, image)
    print(f"绘图完成，结果已保存到: {output_path}")

# 执行函数
draw_polygon_annotations(image_path, label_path, CLASSES)
