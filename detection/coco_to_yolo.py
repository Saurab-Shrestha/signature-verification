from pycocotools.coco import COCO
import os

json_path = "coco_data/annotations/instances_train.json"
images_dir = "coco_data/images/train"
output_dir = "coco_data/labels/train"
os.makedirs(output_dir, exist_ok=True)

coco = COCO(json_path)
category_map = {cat['id']: i for i, cat in enumerate(coco.loadCats(coco.getCatIds()))}

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    yolo_lines = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        cx = x + w / 2
        cy = y + h / 2
        cx /= img_info['width']
        cy /= img_info['height']
        w /= img_info['width']
        h /= img_info['height']
        class_id = category_map[ann['category_id']]
        yolo_lines.append(f"{class_id} {cx} {cy} {w} {h}")

    with open(os.path.join(output_dir, f"{os.path.splitext(img_info['file_name'])[0]}.txt"), 'w') as f:
        f.write('\n'.join(yolo_lines))
