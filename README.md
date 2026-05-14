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
    Mine
    │   ├── config__1.py
    │   ├── config__2.py
    │   └── config__3.py
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


A typical dataset structure may look like this:

```text
mmyolo
├──  data
    ├── mine
        ├── annotations_all_standard
            ├── images/
            │   ├── pic.jpg
            ├── train.json
            ├── test.json
```

## Training

Enter the MMYOLO directory:

```bash
cd mmyolo
```

Train the model using one of the provided configuration files:


```bash
python tools/train.py configs/mine/yolov8_mine.py
```


## Citation

If you use this dataset in your research, please cite our paper, pls contact author for the link of paper at wangyuhao.li@outlook.com.

## License

Please refer to the license information of this repository and the original MMYOLO framework.
