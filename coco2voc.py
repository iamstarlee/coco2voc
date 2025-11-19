import os
import json
from lxml import etree
from PIL import Image

def coco_to_voc_bbox(coco_bbox):
    """COCO bbox: [x, y, w, h] → VOC bbox: xmin, ymin, xmax, ymax"""
    x, y, w, h = coco_bbox
    return int(x), int(y), int(x + w), int(y + h)


def create_voc_xml(img_info, ann_list, categories, output_path):
    annotation = etree.Element("annotation")

    folder = etree.SubElement(annotation, "folder")
    folder.text = "VOC2007"

    filename = etree.SubElement(annotation, "filename")
    filename.text = img_info["file_name"]

    size = etree.SubElement(annotation, "size")
    width = etree.SubElement(size, "width")
    width.text = str(img_info["width"])
    height = etree.SubElement(size, "height")
    height.text = str(img_info["height"])
    depth = etree.SubElement(size, "depth")
    depth.text = "3"

    segmented = etree.SubElement(annotation, "segmented")
    segmented.text = "0"

    # Add objects
    for ann in ann_list:
        obj = etree.SubElement(annotation, "object")
        name = etree.SubElement(obj, "name")
        name.text = categories[ann["category_id"]]

        pose = etree.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = etree.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = etree.SubElement(obj, "difficult")
        difficult.text = "0"

        bbox = etree.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = coco_to_voc_bbox(ann["bbox"])
        etree.SubElement(bbox, "xmin").text = str(xmin)
        etree.SubElement(bbox, "ymin").text = str(ymin)
        etree.SubElement(bbox, "xmax").text = str(xmax)
        etree.SubElement(bbox, "ymax").text = str(ymax)

    tree = etree.ElementTree(annotation)
    tree.write(output_path, pretty_print=True, xml_declaration=False, encoding="utf-8")


def convert_coco_to_voc(coco_json, image_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    # category id → name
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # build image id → annotation list
    img_to_anns = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    for img_info in coco["images"]:
        img_id = img_info["id"]
        filename = os.path.splitext(img_info["file_name"])[0] + ".xml"
        xml_path = os.path.join(output_dir, filename)

        ann_list = img_to_anns[img_id]

        create_voc_xml(img_info, ann_list, categories, xml_path)

        print("Converted:", filename)

    print("Done! All VOC XML files saved to", output_dir)


if __name__ == "__main__":
    coco_json = "/home/lxx/Documents/datasets/coco/annotations/instances_train2017.json"
    image_root = "/home/lxx/Documents/datasets/coco/images/train2017/"
    output_dir = "/home/lxx/Documents/datasets/coco2voc/Annotations/"

    convert_coco_to_voc(coco_json, image_root, output_dir)
