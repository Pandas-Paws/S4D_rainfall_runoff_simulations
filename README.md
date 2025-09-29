
# **S4D Rainfall-Runoff Simulations**

This repository contains the code for the paper *"A Deep State Space Model for Rainfall-Runoff Simulations".* It enables the training and evaluation of three models, including **S4D**, **LSTM**, and **MC-LSTM** (Hoedt et al., 2021)(https://arxiv.org/abs/2101.05186), for rainfall-runoff simulations across 531 CAMELS watersheds.

The code is adapted and organized similarly to [lstm_for_pub](https://github.com/kratzert/lstm_for_pub). As such, many instructions mirror those found in the original repository, with a few modifications specific to this implementation.

---

## **Getting Started**

### **1. Prepare the CAMELS Dataset**
- Download the CAMELS data from [UCAR](https://ral.ucar.edu/solutions/products/camels).
- Place the dataset in the following structure:
  ```
  ./data/basin_dataset_public_v1p2
  ./data/basin_dataset_public_v1p2/camels_attributes_v2.0
  ```

### **2. Update NLDAS Forcings**
- Download updated NLDAS forcings (including daily min and max temperatures) from [HydroShare]([https://www.hydroshare.org/](https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/)).
- Replace the default CAMELS NLDAS forcings, which only include daily mean temperature.

---

## **Training the Models**

### **Run Training Scripts**
To train models, use the script `train_val_global.sh` or `train_pub.sh` for the global (i.e., in-sample) and PUB (i.e., Prediction in Ungauged Basins) setup, respectively, with the following options:
1. **Model Type**: `ssm`, `lstm`, or `mclstm`
2. **Static Input Features**: `static` or `no_static` (include catchment attributes as inputs or not)
3. **Note/Label**: A custom label to distinguish your trained models.

#### **Examples**
- **Train S4D with static inputs under the global setup:**
  ```bash
  ./train_val_global.sh ssm static ensemble
  ```
  Trains an S4D model with 32 inputs (static attributes included). The trained model is labeled as `ensemble`.

- **Train S4D with static inputs under the PUB setup:**
  ```bash
  ./train_pub.sh
  ```
  Trains an S4D model with 32 inputs (static attributes included) with k-fold cross-validation (k=12). No additional argument needed to run the bash file. The number of GPUs and which model to train and test (LSTM or S4D) can be specified inside the bash file.
  
- **Train S4D without static inputs:**
  ```bash
  ./train_val_global.sh ssm no_static nostatic
  ```
  Trains an S4D model with only 5 hydrometeorological inputs (not used in this study). The trained model is labeled as `nostatic`.

### **Hyperparameters**
You can configure hyperparameters for the S4D model inside the `train_val_global.sh` script.

### **Output**
- Trained models are saved in the `runs/` directory.
- Logs are saved in the `reports/` directory with filenames like:
  ```
  global_${model}${static/no_static}${seed}${note}.out # for global setup
  ```
  ```
  pub_${model}.${seed}.${split}.out # for pub setup
  ```

---

## **Testing the Models**

### **Run Test Scripts**
To test trained models, use the script `run_global_parallel.py` with the following options:
1. **Experiment Name**: `global_${model}_${static/no_static}`
2. **Note/Label**: The label specified during training
3. **Epoch**: The number of epochs completed for the model to be tested.

#### **Example**
- **Test S4D with static inputs:**
  ```bash
  python run_global_parallel.py global_ssm_static ensemble 49
  ```
  Tests the trained S4D model (labeled `ensemble`) after 49 epochs under the global setup.

  ```bash
  python run_pub_parallel.py ssm test 49
  ```
  Tests the trained S4D model after 49 epochs under the pub setup, and the saved output .pkl will be labeled as "test".

### **Output**
Test outputs are stored in CSV files in the `./analysis/results/` directory.

---

## **Evaluating the Models**

### **Run Evaluation Scripts**
To evaluate ensemble performance, use the script `main_performance_ensemble_only.py` with the following options:
1. **Experiment Name**: `global_${model}_${static/no_static}`
2. **Note/Label**: The label specified during training
3. **Epoch**: The number of epochs completed for evaluation.

#### **Example**
- **Evaluate S4D with static inputs:**
  ```bash
  python analysis/main_performance_ensemble_only.py global_ssm_static ensemble 49
  ```
  Provides the statistical performance of the trained S4D model (labeled `ensemble`) after 49 epochs.

### **Output**
- Statistical performance tables are stored in the `./analysis/stats/` directory.
- Time series CSV files for all 531 basins are saved in the `./analysis/stats/ssm_metrics/time_series_{note}` folder.

---

## **Key Outputs from Evaluation**
1. **Statistical Performance Tables**: Comprehensive metrics summarizing model performance.
2. **Time Series CSVs**: Basin-specific streamflow predictions for further analysis.
   
---

