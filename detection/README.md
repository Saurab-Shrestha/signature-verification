---
license: agpl-3.0
base_model:
- Ultralytics/YOLOv8
pipeline_tag: object-detection
datasets:
- tech4humans/signature-detection
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
model-index:
- name: tech4humans/yolov8s-signature-detector
  results:
  - task:
      type: object-detection
    dataset:
      type: tech4humans/signature-detection
      name: tech4humans/signature-detection
      split: test
    metrics:
    - type: precision
      value: 0.94499
      name: mAP@0.5
    - type: precision
      value: 0.6735
      name: mAP@0.5:0.95
    - type: precision
      value: 0.947396
      name: precision
    - type: recall
      value: 0.897216
      name: recall
    - type: f1
      value: 0.921623
---

# **YOLOv8s - Handwritten Signature Detection**

This repository presents a YOLOv8s-based model, fine-tuned to detect handwritten signatures in document images.

| Resource                        | Links / Badges                                                                                                                                                                                                                                                                                                                   | Details                                                                                                                                                                 |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Article** | [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md.svg)](https://huggingface.co/blog/samuellimabraz/signature-detection-model) | A detailed community article covering the full development process of the project |
| **Model Files**                 | [![HF Model](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/tech4humans/yolov8s-signature-detector)                                                                                                                                                             | **Available formats:** [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![ONNX](https://img.shields.io/badge/ONNX-005CED.svg?style=flat&logo=ONNX&logoColor=white)](https://onnx.ai/) [![TensorRT](https://img.shields.io/badge/TensorRT-76B900.svg?style=flat&logo=NVIDIA&logoColor=white)](https://developer.nvidia.com/tensorrt) |
| **Dataset – Original**          | [![Roboflow](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/tech-ysdkk/signature-detection-hlx8j)                                                                                                                                                                          | 2,819 document images annotated with signature coordinates                                                                                                           |
| **Dataset – Processed**         | [![HF Dataset](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/tech4humans/signature-detection)                                                                                                                                                  | Augmented and pre-processed version (640px) for model training                                                                                                          |
| **Notebooks – Model Experiments** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wSySw_zwyuv6XSaGmkngI4dwbj-hR4ix) [![W&B Training](https://img.shields.io/badge/W%26B_Training-FFBE00?style=flat&logo=WeightsAndBiases&logoColor=white)](https://api.wandb.ai/links/samuel-lima-tech4humans/30cmrkp8) | Complete training and evaluation pipeline with selection among different architectures (yolo, detr, rt-detr, conditional-detr, yolos)                                        |
| **Notebooks – HP Tuning**       | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wSySw_zwyuv6XSaGmkngI4dwbj-hR4ix) [![W&B HP Tuning](https://img.shields.io/badge/W%26B_HP_Tuning-FFBE00?style=flat&logo=WeightsAndBiases&logoColor=white)](https://api.wandb.ai/links/samuel-lima-tech4humans/31a6zhb1) | Optuna trials for optimizing the precision/recall balance                                                                                                               |
| **Inference Server**            | [![GitHub](https://img.shields.io/badge/Deploy-ffffff?style=for-the-badge&logo=github&logoColor=black)](https://github.com/tech4ai/t4ai-signature-detect-server)                                                                                                                                         | Complete deployment and inference pipeline with Triton Inference Server<br> [![OpenVINO](https://img.shields.io/badge/OpenVINO-00c7fd?style=flat&logo=intel&logoColor=white)](https://docs.openvino.ai/2025/index.html) [![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff)](https://www.docker.com/) [![Triton](https://img.shields.io/badge/Triton-Inference%20Server-76B900?labelColor=black&logo=nvidia)](https://developer.nvidia.com/triton-inference-server) |
| **Live Demo**                   | [![HF Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/tech4humans/signature-detection)                                                                                                                                             | Graphical interface with real-time inference<br> [![Gradio](https://img.shields.io/badge/Gradio-FF5722?style=flat&logo=Gradio&logoColor=white)](https://www.gradio.app/) [![Plotly](https://img.shields.io/badge/PLotly-000000?style=flat&logo=plotly&logoColor=white)](https://plotly.com/python/) |

---

## **Dataset**

<table>
  <tr>
    <td style="text-align: center; padding: 10px;">
      <a href="https://universe.roboflow.com/tech-ysdkk/signature-detection-hlx8j">
        <img src="https://app.roboflow.com/images/download-dataset-badge.svg">
      </a>
    </td>
    <td style="text-align: center; padding: 10px;">
      <a href="https://huggingface.co/datasets/tech4humans/signature-detection">
        <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" alt="Dataset on HF">
      </a>
    </td>
  </tr>
</table>

The training utilized a dataset built from two public datasets: [Tobacco800](https://paperswithcode.com/dataset/tobacco-800) and [signatures-xc8up](https://universe.roboflow.com/roboflow-100/signatures-xc8up), unified and processed in [Roboflow](https://roboflow.com/).

**Dataset Summary:**
- Training: 1,980 images (70%)
- Validation: 420 images (15%)
- Testing: 419 images (15%)
- Format: COCO JSON
- Resolution: 640x640 pixels

![Roboflow Dataset](./assets/roboflow_ds.png)

---

## **Training Process**

The training process involved the following steps:

### 1. **Model Selection:**

Various object detection models were evaluated to identify the best balance between precision, recall, and inference time.


| **Metric**               | [rtdetr-l](https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt) | [yolos-base](https://huggingface.co/hustvl/yolos-base) | [yolos-tiny](https://huggingface.co/hustvl/yolos-tiny) | [conditional-detr-resnet-50](https://huggingface.co/microsoft/conditional-detr-resnet-50) | [detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) | [yolov8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) | [yolov8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) | [yolov8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | [yolov8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) | [yolov8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | [yolo11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | [yolo11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | [yolo11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | [yolo11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | [yolo11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | [yolov10x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt) | [yolov10l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt) | [yolov10b](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt) | [yolov10m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt) | [yolov10s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt) | [yolov10n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt) |
|:---------------------|---------:|-----------:|-----------:|---------------------------:|---------------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|
| **Inference Time - CPU (ms)**  |  583.608 |   1706.49  |   265.346  |                   476.831  |       425.649  | 1259.47 | 871.329 | 401.183 | 216.6   | 110.442 | 1016.68 | 518.147 | 381.652 | 179.792 | 106.656 |  821.183 |  580.767 |  473.109 |  320.12  |  150.076 | **73.8596** |
| **mAP50**               | 0.92709 |   0.901154 |   0.869814 |                   **0.936524** |       0.88885  | 0.794237| 0.800312| 0.875322| 0.874721| 0.816089| 0.667074| 0.707409| 0.809557| 0.835605| 0.813799|  0.681023|  0.726802|  0.789835|  0.787688|  0.663877|  0.734332 |
| **mAP50-95**             |  0.622364 |   0.583569 |   0.469064 |                   0.653321 |       0.579428 | 0.552919| 0.593976| **0.665495**| 0.65457 | 0.623963| 0.482289| 0.499126| 0.600797| 0.638849| 0.617496|  0.474535|  0.522654|  0.578874|  0.581259|  0.473857|  0.552704 |


![Model Selection](./assets/model_selection.png)

#### Highlights:
- **Best mAP50:** `conditional-detr-resnet-50` (**0.936524**)
- **Best mAP50-95:** `yolov8m` (**0.665495**)
- **Fastest Inference Time:** `yolov10n` (**73.8596 ms**)

Detailed experiments are available on [**Weights & Biases**](https://api.wandb.ai/links/samuel-lima-tech4humans/30cmrkp8).

### 2. **Hyperparameter Tuning:**

The YOLOv8s model, which demonstrated a good balance of inference time, precision, and recall, was selected for hyperparameter tuning.

[Optuna](https://optuna.org/) was used for 20 optimization trials.
The hyperparameter tuning used the following parameter configuration:
    
```python
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
    box = trial.suggest_float("box", 3.0, 7.0, step=1.0)
    cls = trial.suggest_float("cls", 0.5, 1.5, step=0.2)
    opt = trial.suggest_categorical("optimizer", ["AdamW", "RMSProp"])
```

Results can be visualized here: [**Hypertuning Experiment**](https://api.wandb.ai/links/samuel-lima-tech4humans/31a6zhb1).  

![Hypertuning Sweep](./assets/sweep.png)

### 3. **Evaluation:**

The models were evaluated on the test set at the end of training in ONNX (CPU) and TensorRT (GPU - T4) formats. Performance metrics included precision, recall, mAP50, and mAP50-95.

![Trials](./assets/trials.png)

#### Results Comparison:

| Metric     | Base Model | Best Trial (#10)  | Difference  |
|------------|------------|-------------------|-------------|
| mAP50      | 87.47%     | **95.75%**        | +8.28%      |
| mAP50-95   | 65.46%     | **66.26%**        | +0.81%      |
| Precision  | **97.23%**      | 95.61%            | -1.63%     |
| Recall     | 76.16%     | **91.21%**        | +15.05%     |
| F1-score   | 85.42%     | **93.36%**        | +7.94%      |

---

## **Results**

After hyperparameter tuning of the YOLOv8s model, the best model achieved the following results on the test set:

- **Precision:** 94.74%
- **Recall:** 89.72%
- **mAP@50:** 94.50%
- **mAP@50-95:** 67.35%
- **Inference Time:**
  - **ONNX Runtime (CPU):** 171.56 ms
  - **TensorRT (GPU - T4):** 7.657 ms  

---

## **How to Use**

The `YOLOv8s` model can be used via CLI or Python code using the [Ultralytics](https://github.com/ultralytics/ultralytics) library. Alternatively, it can be used directly with ONNX Runtime or TensorRT.

The final weights are available in the main directory of the repository:
- [`yolov8s.pt`](yolov8s.pt) (PyTorch format)
- [`yolov8s.onnx`](yolov8s.onnx) (ONNX format)
- [`yolov8s.engine`](yolov8s.engine) (TensorRT format)

### Python Code

- Dependencies

```bash
pip install ultralytics supervision huggingface_hub
```

- Inference 

```python
import cv2
import supervision as sv

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(
  repo_id="tech4humans/yolov8s-signature-detector", 
  filename="yolov8s.pt"
)

model = YOLO(model_path)

image_path = "/path/to/your/image.jpg"
image = cv2.imread(image_path)

results = model(image_path)

detections = sv.Detections.from_ultralytics(results[0])

box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(scene=image, detections=detections)

cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Ensure the paths to the image and model files are correct.


### CLI

- Dependencies

```bash
pip install -U ultralytics "huggingface_hub[cli]"
```

- Inference

```bash
huggingface-cli download tech4humans/yolov8s-signature-detector yolov8s.pt
```

```bash
yolo predict model=yolov8s.pt source=caminho/para/imagem.jpg
```

**Parameters**:
- `model`: Path to the model weights file.
- `source`: Path to the image or directory of images for detection.

### ONNX Runtime

For optimized inference, you can find the inference code using [onnxruntime](https://onnxruntime.ai/docs/) and [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) in the [handler.py](handler.py) file and on the Hugging Face Space [here](https://huggingface.co/spaces/tech4humans/signature-detection).

--- 

## **Demo**

You can explore the model and test real-time inference in the Hugging Face Spaces demo, built with Gradio and ONNXRuntime.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/tech4humans/signature-detection)

---

## 🔗 **Inference with Triton Server**

If you want to deploy this signature detection model in a production environment, check out our inference server repository based on the NVIDIA Triton Inference Server.

<table>
  <tr>
    <td>
      <a href="https://github.com/triton-inference-server/server"><img src="https://img.shields.io/badge/Triton-Inference%20Server-76B900?style=for-the-badge&labelColor=black&logo=nvidia" alt="Triton Badge" /></a>
    </td>
    <td>
      <a href="https://github.com/tech4ai/t4ai-signature-detect-server"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge" /></a>
    </td>
  </tr>
</table>

---

## **Infrastructure**

### Software

The model was trained and tuned using a Jupyter Notebook environment.

- **Operating System:** Ubuntu 22.04
- **Python:** 3.10.12
- **PyTorch:** 2.5.1+cu121
- **Ultralytics:** 8.3.58
- **Roboflow:** 1.1.50
- **Optuna:** 4.1.0
- **ONNX Runtime:** 1.20.1
- **TensorRT:** 10.7.0

### Hardware

Training was performed on a Google Cloud Platform n1-standard-8 instance with the following specifications:

- **CPU:** 8 vCPUs
- **GPU:** NVIDIA Tesla T4

---

## **License**

### Model Weights (Fine-Tuned Model) – **AGPL-3.0**
- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Usage:** The fine-tuned model weights, derived from the YOLOv8 model by Ultralytics, are licensed under AGPL-3.0. This requires that any modifications or derivative works of these model weights also be distributed under AGPL-3.0, and if the model is used as part of a network service, the corresponding source must be made available.

### Code, Training, Deployment, and Data – **Apache 2.0**
- **License:** Apache License 2.0
- **Usage:** All additional materials—including training scripts, deployment code, usage instructions, and associated data—are licensed under the Apache 2.0 license.

For more details, please refer to the full license texts:
- [GNU AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## **Contact and Information**

For further information, questions, or contributions, contact us at **iag@tech4h.com.br**.

<div align="center">
  <p>
    📧 <b>Email:</b> <a href="mailto:iag@tech4h.com.br">iag@tech4h.com.br</a><br>
    🌐 <b>Website:</b> <a href="https://www.tech4.ai/">www.tech4.ai</a><br>
    💼 <b>LinkedIn:</b> <a href="https://www.linkedin.com/company/tech4humans-hyperautomation/">Tech4Humans</a>
  </p>
</div>

## **Author**

<div align="center">
  <table>
    <tr>
      <td align="center" width="140">
        <a href="https://huggingface.co/samuellimabraz">
          <img src="https://avatars.githubusercontent.com/u/115582014?s=400&u=c149baf46c51fdee45ad5344cf1b360236d90d09&v=4" width="120" alt="Samuel Lima"/>
          <h3>Samuel Lima</h3>
        </a>
        <p><i>AI Research Engineer</i></p>
        <p>
          <a href="https://huggingface.co/samuellimabraz">
            <img src="https://img.shields.io/badge/🤗_HuggingFace-samuellimabraz-orange" alt="HuggingFace"/>
          </a>
        </p>
      </td>
      <td width="500">
        <h4>Responsibilities in this Project</h4>
        <ul>
          <li>🔬 Model development and training</li>
          <li>📊 Dataset analysis and processing</li>
          <li>⚙️ Hyperparameter optimization and performance evaluation</li>
          <li>📝 Technical documentation and model card</li>
        </ul>
      </td>
    </tr>
  </table>
</div>

---

<div align="center">
  <p>Developed with 💜 by <a href="https://www.tech4.ai/">Tech4Humans</a></p>
</div>