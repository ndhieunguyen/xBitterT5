<h1 align="center">xBitterT5</h1>

<!-- ![tgmdlm](pics/tgmdlm.png) -->
<p align="center">
        📝 <a href="">Paper</a>｜🤗 <a href="https://huggingface.co/spaces/ndhieunguyen/xBitterT5">Demo</a> | 🚩<a href="https://huggingface.co/ndhieunguyen/xBitterT5">Checkpoints</a>
</p>

This repository is the official implementation of [`xBitterT5`: Identification and explanation of bitter taste peptides with transformer-based architecture ](https://github.com/nhattruongpham/mol-lang-bridge/)

## Abstract
> Abstract here


## News
- 2025.mm.dd: Submitted paper at [Food Chemistry](https://www.sciencedirect.com/journal/food-chemistry) journal.

## Dataset
- BTP640: First proposed in paper [paper](https://www.sciencedirect.com/science/article/pii/S0888754320301725)
- BTP720: First proposed in paper [paper](https://www.sciencedirect.com/science/article/pii/S0308814623019064)

## Data preprocessing
Process the data into two dataframes `train.csv` and `test.csv`, then put them into the same folder. The columns of each dataframe are:
- `No.`: (int) represents the index of the sample
- `sequence`: (str) represents the peptide sequence
- `label`: (int) represents the label of the peptide sequence (0: non-bitter, 1: bitter)

Convert the peptide sequence to SMILES and SELFIES format, and prepare the folds using Stratified K-fold cross-validation algorithm:
```
python3 prepare_data.py
```

## Dependencies and Installation
```
conda env create -f environment.yaml
conda activate xBitterT5
```

## Training
```
bash train.sh
```

## Inferencing
```
bash inference.sh
```

## Interpretation
```
bash interpret.sh
```

## Citation
If you find our research valuable, we kindly ask you to cite it:
```
```