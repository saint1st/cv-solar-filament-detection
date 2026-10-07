## Exploratory Data Analysis

The dataset contains **1,154 images and 8,199 annotations**. Filament areas range from 9 to 37,739 pixels. Only six filament samples are thinner than 5 pixels.

### 1. Filament Size Distribution

| Percentile | Area (pixels) | Description |
|:---:|---:|---|
| **P1** | 209 | Lower bound, excluding extreme outliers |
| **P5** | 317 | Very small structures |
| **P10** | 410 | Small filament threshold |
| **P25** | 670 | Lower quartile (Q1) |
| **P50** | 1,228 | Median filament size |
| **P75** | 2,438 | Upper quartile (Q3) |
| **P90** | 4,685 | Large structures |
| **P95** | 6,947 | Very large structures |
| **P99** | 13,703 | Upper bound, excluding extreme maximums up to 37,739 px |

The wide area distribution motivated the use of dense prediction architectures with large receptive fields, such as U-Net. Because native H-alpha images are `2048 × 2048`, aggressive downscaling (e.g., to `640 × 640`) could destroy fine-scale details, including filament barbs. Preserving high-resolution inputs was therefore essential.

### 2. Instance Distribution and Overlap

![Number of annotations per image](./images/filaments_number_distribution.png)

| Metric | Count | Percentage / Proportion |
|---|---:|---:|
| Images containing ≥2 filaments | 1,051 | 91.1% |
| Images with overlapping masks (Pixel IoU > 0) | 7 | 0.6% |
| Images with touching masks (1–2 px boundary) | 8 | 0.7% |
| Overlapping instance pairs | 7 | 7 / 36,064 pairs |
| Touching instance pairs | 9 | 9 / 36,064 pairs |

### 3. Geometric and Morphological Analysis

We analyzed individual instances using area, aspect ratio, perimeter, solidity, and skeleton length. Skeletonization extracts filament spines and branches, revealing thin structures and secondary barbs that are difficult to preserve during segmentation.

<table>
  <tr>
    <td align="center"><img src="./images/geometry1.png" width="280" alt="Geometric mask analysis 1"></td>
    <td align="center"><img src="./images/geometry2.png" width="280" alt="Geometric mask analysis 2"></td>
    <td align="center"><img src="./images/geometry3.png" width="280" alt="Geometric mask analysis 3"></td>
  </tr>
</table>

| Statistic | Area (px) | Aspect Ratio | Perimeter (px) | Solidity | Skeleton Length (px) |
|---|---:|---:|---:|---:|---:|
| **Count** | 8,199 | 8,199 | 8,199 | 8,199 | 8,199 |
| **Mean** | 2,282.9 | 1.38 | 390.2 | 0.58 | 176.1 |
| **Std** | 2,860.4 | 1.03 | 318.5 | 0.18 | 158.3 |
| **Min** | 6.0 | 0.12 | 6.8 | 0.10 | 3.0 |
| **5%** | 354.0 | 0.33 | 105.6 | 0.28 | 37.0 |
| **25% (Q1)** | 745.5 | 0.66 | 183.3 | 0.45 | 74.0 |
| **50% (Median)** | 1,355.0 | 1.11 | 292.4 | 0.58 | 127.0 |
| **75% (Q3)** | 2,638.0 | 1.79 | 488.1 | 0.71 | 221.0 |
| **95%** | 7,336.7 | 3.34 | 1,026.6 | 0.89 | 487.1 |
| **Max** | 39,145.0 | 12.78 | 3,047.5 | 1.00 | 1,777.0 |

The observed solidity values across samples (0.59–0.74) demonstrate that solar filaments are elongated, irregular, and non-convex rather than solid geometric blobs. These properties favor pixel-level segmentation over bounding-box detection or simple thresholding.

Bivariate log-log analyses further examined the relationships between mask area, aspect ratio, and skeleton length.

- **Area vs. aspect ratio:** The distribution characterizes filament shape and orientation variability.
- **Area vs. skeleton length:** The strong linear relationship in log-log space suggests that area-to-skeleton ratios may help identify morphologically implausible predictions.
- **Pixel-level imbalance:** Foreground occupancy quantifies the dominance of background pixels and informs loss-function selection.

### 4. Pixel-Level Class Imbalance

Filaments occupy approximately **0.30% of image pixels on average** (around 12,600 foreground pixels versus 4.18 million background pixels in a `2048 × 2048` image). Standard Binary Cross-Entropy (BCE) alone can favor trivial all-background predictions.

We therefore considered composite loss functions combining BCE with overlap-based objectives, including **BCE + Dice, BCE + Tversky, and Focal + Dice**.

<table>
  <tr>
    <td align="center"><img src="./images/pixel_occupancy.png" width="400" alt="Pixel occupancy distribution"></td>
    <td align="center"><img src="./images/intencity.png" width="400" alt="Intensity distributions"></td>
  </tr>
</table>

Intensity distributions from 100 sampled images showed substantial overlap between foreground and background pixels. Mean intensity was approximately 83 for the background and 117 for filaments. This overlap limits the effectiveness of global thresholding methods such as Otsu's method and traditional edge detectors such as Canny. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** was therefore incorporated to enhance local contrast before model training.

### 5. Temporal Analysis and Sampling Distribution

Timestamps extracted from filenames (`YYYYMMDDHHMMSSII`) revealed the following temporal structure:

- **Dataset coverage:** 1,154 images across 656 observation days, from January 9, 2011, to August 3, 2022.
- **Median interval:** 1,440 minutes (24 hours) between sequential frames.
- **Closely spaced observations:** 456 frames (39.5%) were captured less than 15 minutes apart.

Temporal proximity can cause information leakage when correlated observations are randomly split between training and validation sets. Validation should therefore group images by observation day or session.

### 6. Covariate Shift and Adversarial Validation

We evaluated train–test photometric differences using adversarial validation.

**Methodology**
1. Extracted image-level mean, standard deviation, and intensity percentiles (P1, P50, P99).
2. Trained a Random Forest classifier to distinguish training samples (`y = 0`) from test samples (`y = 1`).
3. Evaluated the classifier using five-fold cross-validation.

**ROC-AUC: 0.559**

The score indicates limited train–test distinguishability using the selected intensity features. No severe covariate shift was detected by this experiment, although differences in higher-level image characteristics cannot be ruled out.

![Standard deviation distribution](./images/std.png)

### 7. Duplicate Detection and Annotation Fusion

An exhaustive duplicate analysis using byte-level MD5 hashing and timestamp-prefix matching revealed substantial redundancy.

| Finding | Result |
|---|---:|
| Images mapped across exact-duplicate groups | 743 |
| Unique duplicate groups | 296 |
| Original dataset | 1,154 images |
| Final deduplicated dataset | 707 unique frames |
| Filament annotations preserved | 8,199 |

Annotations were fused across duplicate image records, selecting a primary clean frame for each hash group. This reduced the dataset from 1,154 raw images to 707 unique frames while preserving all 8,199 annotations.

To prevent duplicate or temporally correlated observations from crossing validation folds, we generated a unified `group_id` mapping in `train_image_groups.csv`. Using this mapping with **GroupKFold** keeps related samples within the same fold and reduces the risk of overly optimistic validation results.

<table>
  <tr>
    <td align="center"><img src="./images/duplicate1.png" width="280" alt="Duplicate detection"></td>
    <td align="center"><img src="./images/duplicate2.png" width="280" alt="Duplicate grouping"></td>
    <td align="center"><img src="./images/fusion_result1.png" width="280" alt="Annotation fusion"></td>
  </tr>
</table>

The final dataset was serialized as `MAGFiLO_1.0_Annotations_kaggle2026_train_fused_deduplicated.json` for subsequent model training.

### Key Takeaways

- **High-resolution inputs:** Preserve thin filaments and fine morphological details.
- **Morphological complexity:** Motivates dense pixel-level segmentation architectures.
- **Class imbalance:** Requires loss functions that account for foreground overlap.
- **Photometric overlap:** Motivates local contrast enhancement rather than relying on global thresholds.
- **Temporal correlation and duplicates:** Require grouped validation to minimize data leakage.
- **Annotation fusion:** Reduces redundancy while preserving the filament annotations.
