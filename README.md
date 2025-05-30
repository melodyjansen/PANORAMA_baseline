# PANORAMA Challenge – AI in Medical Imaging

This repository is a fork of the official Panorama Challenge baseline: [Original Baseline Repository](https://github.com/DIAGNijmegen/PANORAMA_baseline)

We used this baseline as the foundation for our project. Due to time and computational constraints, we made only minimal changes to the baseline directly. Our main changes were made in a fork of the [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet) repository, where we customized the training procedure. The forked repository can be found here: [Forked nnU-Net](https://github.com/melodyjansen/nnUNet/tree/master)

---

## 🔧 Modifications Overview

### In This Repository
- Forked from the original Panorama baseline.
- Added our own result folder under `src/nnUNet_results/` to include output data and/or logs.
- Adjusted README with added documentation to reflect our setup.

### 🔹 Training Data

The PANORAMA dataset contains approximately 3000 cases split into:

- **2238** training samples (fully accessible)  
- **86** hidden validation cases  
- ~**400** hidden test cases  

Each case includes anonymized CECT scans and annotations (tumor boundaries, labels, etc.). All data is provided under the **CC BY-NC 4.0** license.

For our experiments:  
- We used only the **first batch** of training data: [Batch 1](https://zenodo.org/records/13715870)  
- **6 images were discarded** due to mismatches between CT volumes and labels.  
- Final training was done on **551 cases** after preprocessing with `nnUNetv2_plan_and_preprocess`.  
- Although nnU-Net is typically trained using **5-fold cross-validation**, we only trained on **fold 0** due to limited compute.

---

### 🔹 Our Modified Model

We chose to focus on improving the **segmentation model**, leaving the detection pipeline unaltered.

#### Key Changes:

1. **Resolution Upgrade**  
   Replaced the original **2D low-resolution nnU-Net** with a **3D full-resolution** nnU-Net. While more computationally intensive, 3D models have shown improved performance for volumetric segmentation tasks, particularly in medical imaging (e.g. [Isensee et al., 2021](https://pubmed.ncbi.nlm.nih.gov/33288961/)).

2. **Loss Function**  
   Instead of using only **Dice loss**, we applied a **compound loss**:  
   - 0.5 × Dice loss  
   - 0.5 × TopK loss (with `k = 10%`)  

   This choice was guided by literature showing compound losses improve segmentation performance (e.g. [Ma et al., 2021](https://www.sciencedirect.com/science/article/pii/S1361841521000815)).  
   
   > TopK loss emphasizes difficult voxels by computing cross-entropy over the hardest 10% of predictions.

3. **Training Details**  
   - Model trained for **250 epochs**  
   - Training took approximately **7 hours** on GPU (csedu cluster)  
   - Only **fold 0** used (80% train, 20% validation split)  

---

## 📊 Evaluation Strategy

We were unable to submit our modified container to the Grand Challenge platform due to time and Docker configuration issues. Therefore, we performed our own evaluation using held-out training data.

### Dataset:

- Used **54 images** from the **second batch** of training data (unseen by the model)  
- Due to memory constraints, only **16 images** were ultimately evaluated  

### Metrics:

- **Dice Coefficient** (0–1): measures overlap between predicted and ground truth masks  
- **True Positives (TP)**: correctly predicted lesion voxels  
- **False Positives (FP)**: incorrectly predicted lesion voxels  
- **True Negatives (TN)**: correctly predicted background voxels  
- **False Negatives (FN)**: missed lesion voxels  

The evaluation was conducted across the following labels:  
- PDAC lesion  
- Veins  
- Arteries  
- Pancreas parenchyma  
- Pancreatic duct  
- Common bile duct  


---
# Original Baseline README.md


# Baseline Algorithm for the [PANORAMA challenge](https://panorama.grand-challenge.org/): Pancreatic Cancer Diagnosis - Radiologists Meet AI
This repository contains the baseline algorithm for the [PANORAMA challenge](https://panorama.grand-challenge.org/). Model weights were uploaded to Zenodo and can be downloaded using this link: [https://zenodo.org/records/11160381](https://zenodo.org/records/11160381) The algorithm can be directly accessed and used at [https://grand-challenge.org/algorithms/baseline-panorama/](https://grand-challenge.org/algorithms/baseline-panorama/).

The algorithm is based on the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) (v2) [1] and consists of a two-step approach for pancreatic ductal adenocarcinoma (PDAC) detection on contrast-enhanced CT scans (CECT). First, a low-resolution nnU-Net model with Dice loss is trained to segment the pancreas. Based on this automatic segmentation, a region of interest (ROI) around the pancreas region is cropped from the original input CECT. The cropped ROIs are used to train a high-resolution nnU-Net algorithm for PDAC detection using cross-entropy loss. This process is summarized in the figure.

To build the image container you can use the test_run.sh script. Please note that in order to build the docker using Ubuntu in WLS you first need to enable integration with additional distros in docker desktop for windows (settings -> resources -> WSL integration).

<img src="baseline_pipeline.png" alt="Pipeline of the baseline algorithm for the PANORAMA challenge" style="display: block; margin-bottom: 20px;">

<p align="center">Figure 1: Pipeline of the baseline algorithm for the PANORAMA challenge.</p>

## Model training
Training was performed according to the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) (v2) instructions.
Both nnU-Net algorithms were trained using 5-fold cross-validation. The data split per fold can be found here: [pancreas_segmentation_folds](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/Dataset103_PANORAMA_baseline_Pancreas_Segmentation_folds.json), [PDAC_detection_folds](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/Dataset104_PANORAMA_baseline_PDAC_Detection_folds.json).

### Low-resolution pancreas segmentation model
As a preprocessing step, all images were resampled to (4.5, 4.5, 9.0) spacing, which corresponds to 6x the original median spacing of the training set. This was done using the custom [resample_img](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/data_utils.py#L21) function.
After correctly formatting the dataset according to [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) (v2) instructions and correctly setting all environment variables, example commands for training are:

```
nnUNetv2_plan_and_preprocess -d Dataset103_PANORAMA_baseline_Pancreas_Segmentation --verify_dataset_integrity
nnUNetv2_train -d Dataset103_PANORAMA_baseline_Pancreas_Segmentation 0 3d_fullres --c
nnUNetv2_train -d Dataset103_PANORAMA_baseline_Pancreas_Segmentation 1 3d_fullres --c
nnUNetv2_train -d Dataset103_PANORAMA_baseline_Pancreas_Segmentation 2 3d_fullres --c
nnUNetv2_train -d Dataset103_PANORAMA_baseline_Pancreas_Segmentation 3 3d_fullres --c
nnUNetv2_train -d Dataset103_PANORAMA_baseline_Pancreas_Segmentation 4 3d_fullres --c
```

### High-resolution PDAC detection model
As a preprocessing steps, all images in the training set were cropped according to the pancreas segmentation (ground truth label = 4), using the custom [CropPancreasROI](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/data_utils.py#L53) function. A margin of 100cm x 50cm x 15cm was set according to [previous research](https://pubmed.ncbi.nlm.nih.gov/35053538/) and the anatomical postion of the pancreas [2]. The CE loss was used for training, as this loss is more suitable than Dice for detection tasks [2]. During training network weights were saved every 50 epochs to perform optimal epoch selection. This required a minor alteration to the nnU-Net source code which can be found in [this fork](https://github.com/DIAGNijmegen/nnUNetv2_checkpoints) of the original repository. To install the nnU-Net with this alteration you need to clone the forked repostiory and install the package in editable mode. Furthermore, the [customTrainerCEcheckpoints.py](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/customTrainerCEcheckpoints.py) file must be coppied into the nnunetv2/training/nnUNetTrainer/ folder
```
git clone https://github.com/DIAGNijmegen/nnUNetv2_checkpoints.git
cp /path/to/PANORAMA_baseline/src/customTrainerCEcheckpoints.py nnUnet/nnunetv2/training/nnUNetTrainer/customTrainerCEcheckpoints.py
cd nnUNet
pip install -e .
```

After correctly formatting the dataset according to [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) (v2) instructions and correctly setting all environment variables, example commands for training are:

```
nnUNetv2_plan_and_preprocess -d Dataset104_PANORAMA_baseline_PDAC_Detection --verify_dataset_integrity
nnUNetv2_train -d Dataset104_PANORAMA_baseline_PDAC_Detection 0 -tr nnUNetTrainer_Loss_CE_checkpoints 3d_fullres --c --npz
nnUNetv2_train -d Dataset104_PANORAMA_baseline_PDAC_Detection 1 -tr nnUNetTrainer_Loss_CE_checkpoints 3d_fullres --c --npz
nnUNetv2_train -d Dataset104_PANORAMA_baseline_PDAC_Detection 2 -tr nnUNetTrainer_Loss_CE_checkpoints 3d_fullres --c --npz
nnUNetv2_train -d Dataset104_PANORAMA_baseline_PDAC_Detection 3 -tr nnUNetTrainer_Loss_CE_checkpoints 3d_fullres --c --npz
nnUNetv2_train -d Dataset104_PANORAMA_baseline_PDAC_Detection 4 -tr nnUNetTrainer_Loss_CE_checkpoints 3d_fullres --c --npz
```

#### Best checkpoint selection
By default, the nnU-Net uses the last checkpoint as the final model for each fold. Since we are applying nnU-Net to a detection task, we wrote a custom checkpoint selection method that considers the area under the receiver operating characteristic curve (AUROC) and the average precision (AP) as performance metrics, instead of the Dice score. All metrics were computed using the [picai_eval](https://github.com/DIAGNijmegen/picai_eval) framework.
For each fold, all input validation images (after crop) were saved in fold_x/validation_images. After training a given checkpoint, inference was performed on the validation images using that checkpoint's weights with the following command (example for checkpoint 50, fold 0):
```
nnUNetv2_predict -i /path/to/nnUNet_v2/nnUNet_results/Dataset104_PANORAMA_baseline_PDAC_Detection/nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres/fold_0/validation_images -o /path/to/nnUNet_v2/nnUNet_results/Dataset104_PANORAMA_baseline_PDAC_Detection/nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres/fold_0/validation_check_50 -d 104 -tr nnUNetTrainer_Loss_CE_checkpoints -chk checkpoint_50.pth -f 0 -c 3d_fullres --save_probabilities
```
After inference is done, the performance metrics are computed for the predicted probability maps. This is done using the [picai_eval](https://github.com/DIAGNijmegen/picai_eval) framework in the following way:

```
from picai_eval import Metrics
from picai_eval import evaluate_folder
pm_dir = '/path/to/nnUNet_v2/nnUNet_results/Dataset104_PANORAMA_baseline_PDAC_Detection/nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres/fold_0/validation_check_50'
gt_dir = '/path/to/nnUNet_v2/nnUNet_raw/Dataset104_PANORAMA_baseline_PDAC_Detection/labelsTr'
metrics = evaluate_folder(
        y_det_dir=pm_dir,
        y_true_dir=gt_dir,
        y_true_postprocess_func=lambda lbl: (lbl == 1).astype(int), #considers only the tumor label (1) in the ground truth segmentation
        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0], #extracts candidate lesions from the input probability map
        pred_extensions = ['.npz'])
metrics.save('/path/to/nnUNet_v2/nnUNet_results/Dataset104_PANORAMA_baseline_PDAC_Detection/nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres/fold_0/metrics_check_50.json')
```

After all metrics have been saved to the respective json files, the checkpoint selection can be performed using the provided [notebook](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/checkpoint_selection/Select_Best_Checkpoint_Training.ipynb). The results for the checkpoint selection in this baseline are provided in this [image](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/checkpoint_selection/checkpoint_selection_baseline.png).

#### Final algorithm detection map
The final detection map is obtained by ensambling the best checkpoints for each fold. The nnU-Net output probability map is then converted to a detection map using the [GetFullSizDetectionMap](https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/data_utils.py#L104) function that applies the extract_lesion_candidates method from [report_guided_annotation](https://github.com/DIAGNijmegen/Report-Guided-Annotation) [3]. For more information about the lesion extraction process refer to the documentation in the original repository.



### References:
1. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods. 2021 Feb;18(2):203-211. doi: 10.1038/s41592-020-01008-z. Epub 2020 Dec 7. PMID: 33288961.
2. Alves N, Schuurmans M, Litjens G, Bosma JS, Hermans J, Huisman H. Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography. Cancers (Basel). 2022 Jan 13;14(2):376. doi: 10.3390/cancers14020376. PMID: 35053538; PMCID: PMC8774174.
3. Bosma, J. S., Saha, A., Hosseinzadeh, M., Slootweg, I., de Rooij, M., & Huisman, H. (2023). Semi-supervised Learning with Report-guided Pseudo Labels for Deep Learning-based Prostate Cancer Detection Using Biparametric MRI. Radiology: Artificial Intelligence, doi:10.1148/ryai.230031..
