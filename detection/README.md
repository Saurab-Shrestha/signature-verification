metrics:
- f1
- precision
- recall
library_name: ultralytics
library_version: 8.0.239
inference: false
tags:
- object-detection
- signature-detection
- yolo
- yolov8
- pytorch




The training utilized a dataset built from  public datasets: [Tobacco800](https://paperswithcode.com/dataset/tobacco-800) and [signatures-xc8up](https://universe.roboflow.com/roboflow-100/signatures-xc8up), unified and processed in [Roboflow](https://roboflow.com/) and some private datasets

**Dataset Summary:**
- Format: COCO JSON
- Resolution: 640x640 pixels

![Roboflow Dataset](./assets/roboflow_ds.png)

---

## **How to Use**

The `YOLOv8s` model can be used via CLI or Python code using the [Ultralytics](https://github.com/ultralytics/ultralytics) library. Alternatively, it can be used directly with ONNX Runtime or TensorRT.

The final weights are available in the main directory of the repository:
- [`yolov8s.pt`](yolov8s.pt) (PyTorch format)
- [`yolov8s.onnx`](yolov8s.onnx) (ONNX format)
- [`yolov8s.engine`](yolov8s.engine) (TensorRT format)



## Training command :

'yolo detect train model=yolov8stuned.pt data=signature_coco.yaml epochs=50 imgsz=640 batch=2'
