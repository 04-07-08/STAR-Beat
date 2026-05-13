<div align="justify">

# STAR-Beat: A Spatio-Temporal Attention Recurrent Framework for Lightweight and Interpretable Atrial Fibrillation Detection using Wearable Photoplethysmography

The overall architecture and training strategy of the proposed STAR-Beat framework is as follows. (a) The STAR-Beat Framework: The methodology consists of three stages. Stage I employs CDAE for self-supervised pre-training. Stage II illustrates the multi-task network, which is directly linked to Stage I via Weight Transfer, explicitly inheriting the pretrained encoder parameters to simultaneously perform SQA and Rhythm Classification. Stage III details the comprehensive multi-stage training pipeline with strict patient-level data splitting, where the Architecture Integration highlights the incorporation of the multi-task model into the training flow. (b) Two-Phase Training Strategy: Demonstrates the iterative optimization process from the denoising objective to downstream tasks, highlighting how the weight transfer provides better initialization, faster convergence, and improved generalization. (c) Loss Functions: Formulates the optimization objectives applied in different phases, utilizing MSE Loss for pre-training reconstruction and Focal Loss to address extreme class imbalance during the multi-task fine-tuning phase.

</div>

<img width="1231" height="816" alt="Image" src="https://github.com/user-attachments/assets/ee5311eb-18a8-4478-8001-fd98b616175a" />
