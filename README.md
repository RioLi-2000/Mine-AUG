# Mine-AUG

This repository provides the code and configuration files for the Mine-AUG rock bolt plate detection dataset and data augmentation experiments.

The code is built based on the [MMYOLO](https://github.com/open-mmlab/mmyolo) framework.

## Dataset

The full dataset is available on Hugging Face:

https://huggingface.co/datasets/Wl409/Mine_Rock_Bolt_Plate

## Description

The MRBP dataset contains 289 raw underground mine images and 7,946 annotated rock bolt plate instances.

The dataset is designed for rock bolt plate detection in underground mine environments.

## Annotation Format

Annotations are provided in YOLO format.

Class definition:

```text
0: rock_bolt_plate
```

Each annotation file follows the YOLO format:

```text
class_id x_center y_center width height
```

All bounding box coordinates are normalized between 0 and 1.

## Requirements

This project is based on MMYOLO. Please install MMYOLO and its required dependencies before running the code.

You can refer to the official MMYOLO installation guide:

https://mmyolo.readthedocs.io/

## Project Setup

After installing MMYOLO, copy the files from this repository into the MMYOLO project directory.

### 1. Copy Configuration Files

Put the three Python configuration files in the `config` folder into the MMYOLO `configs` directory.

From:

```text
Mine-AUG/config/
```

To:

```text
mmyolo/configs/
```

The final structure should look like this:

```text
mmyolo/
├── configs/
│   ├── your_config_file_1.py
│   ├── your_config_file_2.py
│   └── your_config_file_3.py
```

### 2. Copy Custom Mine-AUG Module

Put the `mine_aug` folder into the MMYOLO root directory.

From:

```text
Mine-AUG/mine_aug/
```

To:

```text
mmyolo/mine_aug/
```

The final structure should look like this:

```text
mmyolo/
├── configs/
├── tools/
├── mmyolo/
├── mine_aug/
│   ├── ...
│   └── ...
```

## Dataset Preparation

Download the MRBP dataset from Hugging Face:

https://huggingface.co/datasets/Wl409/Mine_Rock_Bolt_Plate

After downloading the dataset, make sure the dataset path in the configuration file is correctly set.

For example, check and modify the `data_root` field in the config file:

```python
data_root = 'path/to/your/dataset/'
```

The dataset should follow the YOLO annotation format.

A typical dataset structure may look like this:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

## Training

Enter the MMYOLO directory:

```bash
cd mmyolo
```

Train the model using one of the provided configuration files:

```bash
python tools/train.py configs/your_config_file.py
```

For example:

```bash
python tools/train.py configs/mine_aug_yolov5.py
```

If the training process is interrupted, you can resume training with:

```bash
python tools/train.py configs/your_config_file.py --resume
```

## Testing

After training, test the model using the trained checkpoint:

```bash
python tools/test.py configs/your_config_file.py work_dirs/your_experiment/epoch_xxx.pth
```

For example:

```bash
python tools/test.py configs/mine_aug_yolov5.py work_dirs/mine_aug_yolov5/epoch_300.pth
```

## Inference

You can run inference on a single image using MMYOLO's demo script:

```bash
python demo/image_demo.py path/to/image.jpg configs/your_config_file.py work_dirs/your_experiment/epoch_xxx.pth --out-dir output
```

For example:

```bash
python demo/image_demo.py demo/demo.jpg configs/mine_aug_yolov5.py work_dirs/mine_aug_yolov5/epoch_300.pth --out-dir output
```

The detection results will be saved in the `output` folder.

## Notes

Before training or testing, please check the following:

1. The MMYOLO environment is installed correctly.
2. The configuration files are placed under `mmyolo/configs/`.
3. The `mine_aug` folder is placed under `mmyolo/mine_aug/`.
4. The dataset path in the config file is correctly set.
5. The annotation files follow YOLO format.

## Citation

If you use this dataset or code in your research, please cite this repository and the dataset page.

## License

Please refer to the license information of this repository and the original MMYOLO framework.
