# Mine-AUG

This repository provides the code and configuration files for the Mine-AUG rock bolt plate detection dataset and augmentation experiments.
The codes built on MMyolo framework.

put the three python file in config folder into mmyolo/configs path

put the folder"mine_aug" in mmyolo/mine_aug
## Dataset

The full dataset is available on Hugging Face:
(https://huggingface.co/datasets/Wl409/Mine_Rock_Bolt_Plate)

## Description

MRBP dataset contains 289 raw underground mine images and 7,946 annotated rock bolt plate instances.

## Annotation format

Annotations are provided in YOLO format.

Class:

```text
0: rock_bolt_plate
