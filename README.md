# GRU4REC-PyTorch
- PyTorch Implementation of the GRU4REC model.
- Original paper: [Session-based Recommendations with Recurrent Neural Networks(ICLR 2016)](https://arxiv.org/pdf/1511.06939.pdf)
- This code is based on [pyGRU4REC](https://github.com/yhs-968/pyGRU4REC) that is implemented by Younghun Song (yhs-968) and [original Theano code written by the authors of the GRU4REC paper](https://github.com/hidasib/GRU4Rec)
- In this version, I added TOP1-max, BPR-max, restruct project, init weight and load model to evaluate.

## Requirements
- Pytorch 0.4.1
- Python 3.5
- pandas
- numpy 1.14.5

## Usage

### Pre processing data
You need to run preprocessing.py to obtain training data and testing data. In this code, I created the mode that extend x percent of last training data to improve result.

The format of data is:
- Filenames
    - Training set should be named as `rsc15_train_full.txt`
    - Test set should be named as `rsc15_test.txt`
- Contents
    - `rsc15_train_full.txt`, `rsc15_test.txt` should be the tsv files that stores the pandas dataframes that satisfy the following requirements(without headers):
        - The 1st column of the file should be the integer Session IDs
        - The 2nd column of the file should be the integer Item IDs
        - The 3rd column of the file should be the Timestamps
        
### Training and Testing
The project have a structure as below:

```bash
├── GRU4REC-pytorch
│   ├── checkpoint
│   ├── data
│   │    ├── preprocessed_data
│   │    │    ├── rsc15_test.txt
│   │    │    ├── rsc15_train_full.txt
│   │    │    ├── rsc15_train_tr.txt
│   │    │    ├── rsc15_train_valid.txt
│   │    ├── raw_data
│   │    │    ├── yoochoose-clicks.dat
│   ├── lib
│   ├── main.py
│   ├── preprocessing.py
│   ├── tool.py
```

I use tool.py to get 1/8 last yoochoose-clicks.dat

In GRU4REC-pytorch

Training 
```bash
python3 main.py
```

Testing
```bash
python3 main.py --eval --load_model checkpoint/11081713/model_00004.pt
```
 
Logs
```bash
PARAMETER----------
BATCH_SIZE=50
CHECKPOINT_DIR=checkpoint/11081713
CUDA=True
DATA_FOLDER=data/preprocessed_data_FULL
DROPOUT_HIDDEN=0.5
DROPOUT_INPUT=0
EMBEDDING_DIM=-1
EPS=1e-06
FINAL_ACT=tanh
HIDDEN_SIZE=100
IS_EVAL=False
LOAD_MODEL=None
LOSS_TYPE=BPR
LR=0.01
MODEL_NAME=GRU4REC
MOMENTUM=0
N_EPOCHS=10
NUM_LAYERS=1
OPTIMIZER_TYPE=Adagrad
SAVE_DIR=models
SEED=7
SIGMA=None
TEST_DATA=rsc15_test.txt
TIME_SORT=False
TRAIN_DATA=rsc15_train_full.txt
VALID_DATA=rsc15_test.txt
WEIGHT_DECAY=0
-------------------
/usr/local/lib/python3.5/dist-packages/torch/nn/modules/rnn.py:38: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
  "num_layers={}".format(dropout, num_layers))
Epoch: 0, loss: 0.34, recall: 0.57, mrr: 0.24, time: 2962.893368244171
Save model as checkpoint/11081713/model_00000.pt
Epoch: 1, loss: 0.32, recall: 0.60, mrr: 0.25, time: 2961.867926120758
Save model as checkpoint/11081713/model_00001.pt
Epoch: 2, loss: 0.32, recall: 0.60, mrr: 0.25, time: 2961.7242798805237
Save model as checkpoint/11081713/model_00002.pt
Epoch: 3, loss: 0.31, recall: 0.61, mrr: 0.25, time: 2960.437418460846
Save model as checkpoint/11081713/model_00003.pt
Epoch: 4, loss: 0.31, recall: 0.61, mrr: 0.25, time: 2958.951169013977
Save model as checkpoint/11081713/model_00004.pt
Epoch: 5, loss: 0.31, recall: 0.61, mrr: 0.25, time: 2961.2622771263123
Save model as checkpoint/11081713/model_00005.pt
Epoch: 6, loss: 0.31, recall: 0.61, mrr: 0.25, time: 2961.8305492401123
Save model as checkpoint/11081713/model_00006.pt
Epoch: 7, loss: 0.31, recall: 0.61, mrr: 0.25, time: 2961.3042261600494
Save model as checkpoint/11081713/model_00007.pt
Epoch: 8, loss: 0.30, recall: 0.61, mrr: 0.25, time: 2961.863669157028
Save model as checkpoint/11081713/model_00008.pt
Epoch: 9, loss: 0.30, recall: 0.61, mrr: 0.25, time: 2962.760101079941
Save model as checkpoint/11081713/model_00009.pt
```

## Results

With loss function BPR and full dataset, I get 0.61 in recall@20 and 0.25 in mrr@20 

With loss function TOP1 and full dataset, I get 0.62 in recall@20 and 0.26 in mrr@20
 
