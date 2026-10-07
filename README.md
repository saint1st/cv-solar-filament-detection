# cv-solar-filament-detection
A machine learning pipeline for solar filament detection, built as part of a Kaggle competition

# Automated Segmentation of Solar Filaments

## Overview
This repository contains the source code and machine learning pipeline developed for the Kaggle challenge on automated solar filament segmentation. The objective of this project is to generate highly accurate, pixel-level segmentation masks from H-Alpha solar observations. 

Accurate detection of solar filaments is critical for space weather research, as they are at the core of solar eruptions such as Coronal Mass Ejections (CMEs) and solar flares.

## Dataset: MAGFiLO
The models are trained and evaluated on **MAGFiLO** (Manually Annotated GONG Filaments from H-Alpha Observations). This dataset provides ground-truth segmentation masks created by expert human annotators. 

**Key Challenges Addressed:**
- **Fine-scale structures:** Capturing thread-like features (barbs) that extend from the main filament body.
- **Background noise:** Distinguishing filament material from imaging artifacts inherent to ground-based observatories.
- **Structural continuity:** Preventing fragmented segmentations and ensuring contiguous physical representations of filament morphology.

## Methodology
> **Note for the reader:** Briefly summarize your specific approach here.

- **Architecture:** Implemented a [insert model name, e.g., U-Net / YOLOv8] tailored for semantic segmentation of fine-scale structures.
- **Preprocessing:** Applied [insert techniques] to suppress background noise and enhance image quality.
- **Post-processing:** Leveraged [insert techniques, e.g., morphological operations] to enforce structural continuity and minimize over-merging.

## Evaluation Metrics
The pipeline is evaluated based on the competition's strict criteria:
- **Panoptic Quality (PQ):** The primary metric measuring both segmentation accuracy and instance recognition.
- **Dice Score & IoU:** Utilized (`torchmetrics.segmentation.DiceScore`) to quantify the exact overlap between predicted and ground-truth masks.
- **Fragmentation Penalties:** The evaluation penalizes one-to-many and many-to-one mapping errors.

## Repository Structure
```text
├── data/                   # Directory for MAGFiLO test/train datasets
├── notebooks/              # Jupyter notebooks illustrating the entire pipeline
├── src/                    # Source code for training and inference
├── requirements.txt        # Utilized packages and their corresponding versions
├── technical_report.pdf    # Detailed 4-page technical report of the methodology
└── README.md
