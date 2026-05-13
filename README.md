<div align="justify">

# STAR-Beat: A Spatio-Temporal Attention Recurrent Framework for Lightweight and Interpretable Atrial Fibrillation Detection using Wearable Photoplethysmography
Official code repository for the paper: "STAR-Beat: A Spatio-Temporal Attention Recurrent Framework for Lightweight and Interpretable Atrial Fibrillation Detection using Wearable Photoplethysmography".

# 📖 Abstract

Real-time Atrial Fibrillation (AF) detection using Wearable Photoplethysmography (PPG) is highly constrained by severe motion artifacts (MA), lack of rhythm-dependency extraction, and strict hardware limitations of edge devices. Furthermore, the "black-box" nature of deep learning models limits clinical trust.

To address these gaps, we propose STAR-Beat, an ultra-lightweight, temporally-aware, and highly interpretable multi-task AF detection framework.
Morphological Prior Extraction: Utilizes a Convolutional Denoising Autoencoder (CDAE).
Spatio-Temporal Awareness: Integrates Squeeze-and-Excitation (SE) channel attention with a Bidirectional GRU (BiGRU) to capture long-range irregular rhythm dependencies.
Multi-Task Learning & Focal Loss: Simultaneously assesses Signal Quality (SQA) and Cardiac Rhythm, effectively decoupling noise from physiological features.
Edge-Oriented Compression: Achieves an ultra-low inference delay (1.885ms) with only 8.6% of baseline parameters.
Clinical Interpretability: Validated via UMAP, Grad-CAM, Integrated Gradients, and Occlusion Sensitivity.

</div>

<p align="center">
<img width="1231" height="816" alt="Image" src="https://github.com/user-attachments/assets/ee5311eb-18a8-4478-8001-fd98b616175a" />
<br>
<em>Figure 1: Overall architecture and training strategy of the proposed STAR-Beat framework.</em>
</p>

<div align="justify">

# 🗂️ Repository Structure

The codebase is organized into modular scripts for training, evaluation, ablation studies, and clinical interpretability visualization.
## Core Training & Evaluation
deep_1_improved.py: Main script for CDAE pretraining and multi-task fine-tuning (Stanford dataset).

evaluation_improved.py: Comprehensive evaluation script featuring dynamic threshold optimization (Max F1) and ROC metrics.

mimic_train_improved_pretrain.py: Training script tailored for the noisy MIMIC PERform dataset.

mimic_train_evaluation_improved.py: Evaluation on MIMIC with advanced IQR-based Signal Quality Assessment (SQA).

## Ablation & Model Lightweighting
ablation_experiement_se_bigru.py: Configurable script to toggle Pretraining, Multi-task, SE, and BiGRU modules.

ablation_evaluation.py: Evaluates specific ablation variants.

deep_1_lite_stage_validation.py: Progressive structural compression script (Standard Conv -> SeparableConv -> GAP substitution -> Channel Pruning).
## Interpretability
visualize_mimic_1_actual.py: Generates UMAP latent space projections and Post-hoc Interpretability maps (Grad-CAM, Integrated Gradients, Occlusion Sensitivity).

# 🚀 Getting Started

## 1. Requirements
Install the required dependencies: pip install numpy pandas scipy scikit-learn tensorflow matplotlib umap-learn
The detailed information about version is as follows: numpy==1.26.4, pandas==2.3.3, scipy==1.15.3, scikit-learn==1.7.2, tensorflow==2.10.1, matplotlib==3.10.8, umap-learn==0.5.11

## 2. Data Preparation
The model is validated on two public datasets:

1.Stanford PPG Dataset: https://www.synapse.org/Synapse:syn21985690/files/

2.MIMIC PERform Dataset: https://zenodo.org/records/15906524

Ensure your data is extracted and formatted correctly. Modify the DATA_PATH and EXTRACTED_DIR variables inside the scripts to point to your local .npz or .csv directories.

# 💻 Usage
## Step 1: Model Training (End-to-End)
To train the improved STAR-Beat model from scratch (including simulated CDAE pre-training and real-data fine-tuning):**python deep_1_improved.py**.For MIMIC dataset, use **mimic_train_improved_pretrain.py**.

## Step 2: Evaluation
Evaluate the trained model and find the optimal decision threshold dynamically: **evaluation_improved.py**/**mimic_train_evaluation_improved**

## Step 3: Ablation Studies
To verify the contribution of the SE attention, BiGRU, pretrainig, or Multi-task architecture, modify the boolean flags at the top of the ablation script and run:**python ablation_experiement_se_bigru.py** **python ablation_evaluation.py**

## Step 4: Comparision experiment
In order to comprehensively verify the advancement and effectiveness of the STAR-Beat model in the detection of atrial fibrillation, four representative methods are carefully selected as baseline models for multi-dimensional comparison.
<div align="center">

**TABLE I. THE EVALUATION RESULTS OF EACH MODEL ON THE TEST SET**

| Model | TPR | PPV | F1 | AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **VGG16** | 0.917 | 0.496 | 0.644 | 0.886 |
| **PPGVGGNet** | 0.828 | 0.539 | 0.653 | 0.890 |
| **ResNeXt** | **0.990** | 0.581 | 0.732 | 0.987 |
| **BayesBeat** | 0.986 | 0.894 | 0.938 | 0.995 |
| **DeepBeat** | 0.980 | 0.939 | 0.959 | 0.920 |
| **STAR-Beat (proposed)** | 0.954 | **0.970** | **0.962** | **0.998** |

</div>

## Step 5: Lightweighting
Test the 3-stage ablation compression strategy for wearable devices. Modify ABLATION_STEP =  1, 2, or 3 in the script to test different compression limits: **python deep_1_lite_stage_validation.py**

## Step 6: Clinical Interpretability Visualizations
Generate physiological heatmaps to verify the mathematical and medical logic of the model: **python visualize_mimic_1_actual.py**

This script will output: **UMAP** Latent Space Clustering and Saliency Maps (**Grad-CAM, IG, Occlusion**).

# 🔍 Model Explainability
STAR-Beat bridges the trust gap in deep learning by providing explicit physiological transparency. Our interpretability module proves that the model focuses on erratic rhythms, variable amplitudes, and morphological distortions (pathological hemodynamic characteristics), rather than fitting data biases.
<p align="center">
<img src="https://via.placeholder.com/800x400.png?text=Replace+with+Fig.+2+from+the+paper" alt="Interpretability Visualizations">
<br>
<em>Fig. 2.	Visualization of the deep feature space and model interpretability.</em>
</p>
