# T-3DGS: Removing Transient Objects for 3D Scene Reconstruction

## [Project Page](https://transient-3dgs.github.io/) | [Paper](https://arxiv.org/abs/2412.00155)

## Abstract
We propose a novel framework to remove transient objects
from input videos for 3D scene reconstruction using Gaussian Splatting. Our framework consists of the following
steps. In the first step, we propose an unsupervised training
strategy for a classification network to distinguish between
transient objects and static scene parts based on their different training behavior inside the 3D Gaussian Splatting reconstruction. In the second step, we improve the boundary quality and stability of the detected transients by combining our results from the first step with an off-the-shelf segmentation method. We also propose a simple and effective strategy
to track objects in the input video forward and backward in
time. Our results show an improvement over the current
state of the art in existing sparsely captured datasets and
significant improvements in a newly proposed densely captured (video) dataset.

## Overview
This repository implements Transient Mask Predictor (TMP), a solution for handling transient objects in 3D scene reconstruction. For mask refinement functionality (TMR), please refer to our [companion repository](https://github.com/Vadim200116/AutoVidSeg).


## Key Features
- **Automatic Detection of Transient Objects:** Integrate transient object removal seamlessly into the 3D reconstruction pipeline.
- **Two-Stage Pipeline:** Combines TMP and TMR for enhanced mask prediction and refinement.
- **Docker Support:** Simplifies deployment and setup across different environments.


## Installation

The installation process aligns with the original [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) project, with additional dependencies specified in `environment.yml`. We also provide a `Dockerfile` for containerized setups.

## Run Experiments

By default, the following features are enabled:

- **Transient Mask Prediction (TMP)**
- **Mask Dilation**
- **Consistency Loss**
- **Depth Regularization**

### Training the Model

To start training with default settings:

```bash
python train.py -s [path to dataset]
```

### Customizing Training Options
To disable specific features, use the following flags:
- **Disable Transient Mask Predictor (TMP):**

```bash
python train.py -s [path to dataset] --disable_transient
```

- **Disable Mask Dilation:**

```bash
python train.py -s [path to dataset] --disable_dilate
```

- **Disable Consistency Loss:**

```bash
python train.py -s [path to dataset] --disable_consistency
```
- **Disable Depth Regularization:**

```bash
python train.py -s [path to dataset] --lambda_tv 0
```
### Training With Precomputed Masks

```bash
python train.py -s [path to dataset] --masks [path to masks] --disable_transient
```
- Masks should be in `.png` format.
- Masks can have any naming format.
- Images and masks are matched based on their positions in the [nasorted](https://www.geeksforgeeks.org/python-natsorted-function/) lists of image filenames and mask filenames.
- It is recommended to slightly dilate your masks to account for potential inaccuracies. Use the `--mask_dilate` flag (default is 5).

### Bechmarking

#### Running TMP Benchmark
To run all experiments without TMR:

```bash
bash examples/tmp_benchmark.sh
```
This script will initiate the training and evaluation processes for the TMP without mask refinement.

#### Mask Refinement with TMR
To refine transient masks using TMR, follow these steps:

1. **Prepare TMR input**

Run the preparation script:
```bash
bash examples/prepare_tmr_input.sh
```
This script performs the following actions:

- **Reformats Images**: Converts images to the format required by SAM2.

- **Extracts Transient Masks and Differences**: Retrieves transient masks and difference images from your T-3DGS checkpoint (default iteration is 7000).


2. **Run TMR**

- **Follow Instructions:** Visit the [TMR Repository](https://github.com/Vadim200116/AutoVidSeg?tab=readme-ov-file#promptable-mode) for detailed instructions.
- **Execute Refinement Script:** Use the [provided script](https://github.com/Vadim200116/AutoVidSeg/blob/main/examples/tmr_benchmark.sh) in the TMR repository to perform mask refinement.

3. **Final Training with Refined Masks**
After obtaining refined masks from TMR, run the following script to train the model with these masks:

```bash
bash examples/tmr_benchmark.sh
```

## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{pryadilshchikov2024t3dgsremovingtransientobjects,
      title={T-3DGS: Removing Transient Objects for 3D Scene Reconstruction}, 
      author={Vadim Pryadilshchikov and Alexander Markin and Artem Komarichev and Ruslan Rakhimov and Peter Wonka and Evgeny Burnaev},
      year={2024},
      eprint={2412.00155},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00155}, 
}
```

## Acknowledgments
Our code is based on the official [implementation](https://github.com/graphdeco-inria/gaussian-splatting) of 3D Gaussian Splatting (3DGS).