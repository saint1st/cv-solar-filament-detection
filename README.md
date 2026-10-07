# cv-solar-filament-detection
A machine learning pipeline for solar filament detection, built as part of a Kaggle competition

# Solar Filaments Segmentation Challenge

Solar filaments are dense, relatively cool "clouds" of solar material (plasma) that are suspended above the Sun's surface by powerful magnetic field lines.

1. The Visual Appearance
Even though they are extremely hot, filaments are cooler than the solar surface (photosphere) below them. When viewed through specific wavelengths (like the H-Alpha observations in the MAGFiLO dataset), they absorb the light from below and appear as dark, thread-like streaks or ribbons across the Sun's bright disk. If you view a filament on the edge of the Sun against the blackness of space, it glows bright red—in that position, it is called a solar prominence.

2. The Danger (Space Weather)
Filaments are highly volatile. When the magnetic fields holding them become unstable, they can erupt, violently throwing billions of tons of plasma into space. This is known as a Coronal Mass Ejection (CME). If an Earth-directed CME hits our magnetic field, it can:

 - Overload and destroy electric power grids.
 - Disrupt GPS navigation and satellite communications.
 - Expose astronauts and passengers on high-altitude polar flights to dangerous levels of radiation.
## Overview
This repository contains the source code and machine learning pipeline developed for the Kaggle challenge on [solar filament segmentation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview). 

The objective of this project is to generate highly accurate, pixel-level segmentation masks from H-Alpha solar observations. 

## Dataset: MAGFiLO
The models are trained and evaluated on **MAGFiLO** (Manually Annotated GONG Filaments from H-Alpha Observations). 

**Image Specifications:**
- **Format:** 2048 × 2048 pixels, 8-bit JPEG.
- **Color Space:** Grayscale (not to be processed as RGB).
- **Naming Convention:** Files are named sequentially as `YYYYMMDDHHMMSSII` (e.g., `20260901165702Bh.jpeg` encodes the capture date, time, and the Big Bear observatory code).

![Training images examples](./images/1.png)
![Training images examples](./images/2.png)

![Training images examples](./images/duplicate1.png)
![Training images examples](./images/duplicate2.png)

![Training images examples](./images/fusion_result1.png)


**Annotation Details:**
- **Format:** Ground-truth segmentations are stored as polygons in Run-Length Encoding (RLE) for storage efficiency (lossless conversion to binary masks).
- **COCO Compatibility:** Annotations are structured in a COCO-style format, allowing seamless integration with the `pycocotools` library.
- **Multiple Annotators:** A single H-Alpha observation may contain multiple filaments, and the same image might be evaluated by different annotators independently. These instances are treated as distinct image records in the dataset.(This redundancy introduces overlapping and duplicate annotations into the training set)




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
