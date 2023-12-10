# Project Name

隱私保護與醫學數據標準化競賽：解碼臨床病例、讓數據說故事

## Table of Contents

- [Environment](#Environment)

- [Installation](#Installation)

- [Dataset](#Dataset)

- [Usage](#Usage)

  


## Environment

- 作業系統：Ubuntu 20.04.6 LTS
- 程式語言：Python 3.9.17
- CPU規格：Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
- GPU規格：NVIDIA GeForce RTX 3090 Ti
- CUDA版本：12.3

## Installation

```
conda env create -f ./environment.yml
conda activate knn-training
```

## Dataset

第一部份訓練集 + 第二部份訓練集 + 第一部份驗證集

## Usage

-  Train the model

```
python train.py \
  --model_name 'pythia-1b-deduped' \
  --train_file_dir './AICUP_datasets/all-datasets/train_datasets' \
  --trainer_output_file './AICUP_datasets/all-datasets/format_file.tsv' \
  --train_anno_path './AICUP_datasets/all-datasets/answer.txt' \
  --epochs 20 \
  --batch_size 4 \
  --lora True\
  --wandblog False
```

 -  Test the model

```
python train.py \
  --model_name 'pythia-1b-deduped' \
  --valid_file_dir './AICUP_datasets/all-datasets/train_datasets' \
  --valid_output_file './AICUP_datasets/opendid_test/valid_format_file.tsv' \
  --batch_size 16 \
  --lora True
```

> ### Parameters

General settings:

- `--batch_size`：設定batch size
- `--lora`：設定是否啟用 LoRA

For trainning:

- `--model_name`：設定模型
- `--train_file_dir`：指定訓練集的路徑
- `--trainer_output_file`：設定訓練輸出文件的路徑
- `--train_anno_path`：指定訓練集答案文件的路徑
- `--epochs`：設定訓練時的迭代次數
- `--wandblog`: 設定是否啟用Weights & Biases 工具

For prediction:

- `--valid_file_dir`：指定測試集的路徑。
- `--valid_output_file`：設定測試集格式化輸出文件的路徑

