<div align="justify">

# STAR-Beat: A Spatio-Temporal Attention Recurrent Framework for Lightweight and Interpretable Atrial Fibrillation Detection using Wearable Photoplethysmography
Official code repository for the paper: "STAR-Beat: A Spatio-Temporal Attention Recurrent Framework for Lightweight and Interpretable Atrial Fibrillation Detection using Wearable Photoplethysmography".

# 📖 Abstract

Real-time Atrial Fibrillation (AF) detection using Wearable Photoplethysmography (PPG) is highly constrained by severe motion artifacts (MA), lack of rhythm-dependency extraction, and strict hardware limitations of edge devices. Furthermore, the "black-box" nature of deep learning models limits clinical trust.

To address these gaps, we propose STAR-Beat, an ultra-lightweight, temporally-aware, and highly interpretable multi-task AF detection framework.

&emsp;&emsp;1) Morphological Prior Extraction: Utilizes a Convolutional Denoising Autoencoder (CDAE).

&emsp;&emsp;2) Spatio-Temporal Awareness: Integrates Squeeze-and-Excitation (SE) channel attention with a Bidirectional GRU (BiGRU) to capture long-range irregular rhythm dependencies.

&emsp;&emsp;3) Multi-Task Learning & Focal Loss: Simultaneously assesses Signal Quality (SQA) and Cardiac Rhythm, effectively decoupling noise from physiological features.

&emsp;&emsp;4) Clinical Interpretability: Validated via UMAP, Grad-CAM, Integrated Gradients, and Occlusion Sensitivity.

&emsp;&emsp;5) Edge-Oriented Compression: Achieves an ultra-low inference delay (1.885ms) with only 8.6% of baseline parameters.

The overall architecture of the model is shown in Fig. 1.

</div>

<p align="center">
<img width="1231" height="816" alt="Image" src="https://github.com/user-attachments/assets/f6a485ad-9d47-49a0-baae-2a3ec748d841" /
<br>
<em>Figure 1: Overall architecture and training strategy of the proposed STAR-Beat framework.(a) The STAR-Beat Framework: The methodology consists of three stages. Stage I employs CDAE for self-supervised pre-training. Stage II illustrates the multi-task network, which is directly linked to Stage I via Weight Transfer, explicitly inheriting the pretrained encoder parameters to simultaneously perform SQA and Rhythm Classification. Stage III details the comprehensive multi-stage training pipeline with strict patient-level data splitting, where the Architecture Integration highlights the incorporation of the multi-task model into the training flow.(b) Two-Phase Training Strategy: Demonstrates the iterative optimization process from the denoising objective to downstream tasks, highlighting how the weight transfer provides better initialization, faster convergence, and improved generalization.(c) Loss Functions: Formulates the optimization objectives applied in different phases, utilizing MSE Loss for pre-training reconstruction and Focal Loss to address extreme class imbalance during the multi-task fine-tuning phase.</em>
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

&emsp;&emsp;1.Stanford PPG Dataset: https://www.synapse.org/Synapse:syn21985690/files/

&emsp;&emsp;2.MIMIC PERform Dataset: https://zenodo.org/records/15906524

Ensure your data is extracted and formatted correctly. Modify the DATA_PATH and EXTRACTED_DIR variables inside the scripts to point to your local .npz or .csv directories.

# 💻 Usage
## Step 1: Model Training 
To train the improved STAR-Beat model from scratch (including simulated CDAE pre-training and real-data fine-tuning), please run **python deep_1_improved.py**. For MIMIC dataset, use **mimic_train_improved_pretrain.py**.

## Step 2: Evaluation
Evaluate the trained model and find the optimal decision threshold dynamically: **evaluation_improved.py**( For Stanford Public Dataset ) / **mimic_train_evaluation_improved**( For MIMIC PERForm )

The performance on Stanford Public Dataset and MIMIC PERForm is shown as TABLE I.
<div align="center">

**TABLE I. STAR-BEAT MODEL PERFORMANCE ON PUBLIC DATASETS**

| Dataset | Precision (PPV) | Recall (TPR) | F1-Score | AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **Stanford Public** | 0.970 | 0.954 | 0.962 | 0.998 |
| **MIMIC PERform** | 0.964 | 0.928 | 0.946 | 0.986 |

</div>


## Step 3: Ablation Studies
To verify the contribution of the SE attention, BiGRU, pretrainig, or Multi-task architecture, modify the boolean flags at the top of the ablation script and run: **python ablation_experiement_se_bigru.py** 

The ablation study results are displayed in TABLE II.

<div align="center">

**TABLE II. EVALUATION OF THE PERFORMANCE OF EACH MODEL VARIANT IN THE ABLATION EXPERIMENT**

| Model Variant | TPR | PPV | F1 | AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **STAR-Beat-singletask** | 0.929 | 0.318 | 0.473 | 0.706 |
| **STAR-Beat-nopretrain** | 0.907 | **0.980** | 0.942 | 0.997 |
| **STAR-Beat-nose**       | 0.747 | 0.334 | 0.462 | 0.810 |
| **STAR-Beat-nobigru**    | 0.930 | 0.970 | 0.949 | 0.997 |
| **STAR-Beat**            | **0.954** | 0.970 | **0.962** | **0.998** |

</div>

## Step 4: Comparision experiment
In order to comprehensively verify the advancement and effectiveness of the STAR-Beat model in the detection of atrial fibrillation, four representative methods are carefully selected as baseline models for multi-dimensional comparison, and the performance on each model is shown in TABLE III.
<div align="center">

**TABLE III. THE EVALUATION RESULTS OF EACH MODEL ON THE TEST SET**

| Model | TPR | PPV | F1 | AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **VGG16** | 0.917 | 0.496 | 0.644 | 0.886 |
| **PPGVGGNet** | 0.828 | 0.539 | 0.653 | 0.890 |
| **ResNeXt** | **0.990** | 0.581 | 0.732 | 0.987 |
| **BayesBeat** | 0.986 | 0.894 | 0.938 | 0.995 |
| **DeepBeat** | 0.980 | 0.939 | 0.959 | 0.920 |
| **STAR-Beat (proposed)** | 0.954 | **0.970** | **0.962** | **0.998** |

</div>

## Step 5: Clinical Interpretability Visualizations
Generate physiological heatmaps to verify the mathematical and medical logic of the model: **python visualize_mimic_1_actual.py**

This script will output: **UMAP** Latent Space Clustering and Saliency Maps (**Grad-CAM, IG, Occlusion**), which is shown in Fig.2.

<p align="center">
<img width="9424" height="4363" alt="Image" src="https://github.com/user-attachments/assets/49e20170-a0ee-4da5-ad79-7c13d6b91768" />
<br>
<em>Fig. 2.	Visualization of the deep feature space and model interpretability. (a) UMAP visualization of the extracted deep features, demonstrating high separability between classes. (b-g) Temporal importance score mapping for Atrial Fibrillation (left column) and Normal Sinus Rhythm (right column) using three interpretability methods: Class Activation Mapping, Integrated Gradients , and Occlusion Sensitivity. The color map transitions from cool to warm colors, indicating increasing contribution to the model's final prediction.</em>
</p>

## Step 6: Lightweighting
Test the 3-stage ablation compression strategy for wearable devices. Modify ABLATION_STEP =  1, 2, or 3 in the script to test different compression limits: **python deep_1_lite_stage_validation.py**

The variantion of parameters and performance during the compression process is shown in TABLE IV.

<div align="center">

**TABLE IV. COMPARISON OF MODEL COMPRESSION EFFECTS**

| Step | Params | Retention ratio | Latency (ms) | F1 score |
| :--- | :---: | :---: | :---: | :---: |
| **Step0** | 1,516,149 | 100.0% | 11.492 | 0.9622 |
| **Step1** | 569,653 | 37.6% | 4.615 | 0.8774 |
| **Step2** | 462,389 | 30.5% | 2.177 | 0.8517 |
| **Step3** | 130,557 | 8.6% | 1.885 | 0.8306 |

</div>
