# GRU4REC-PyTorch
- PyTorch Implementation of the GRU4REC model.
- Original paper: [Session-based Recommendations with Recurrent Neural Networks(ICLR 2016)](https://arxiv.org/pdf/1511.06939.pdf)
- Extension over the Original paper: [Recurrent Neural Networks with Top-k Gains for Session-based
Recommendations(CIKM 2018)](https://arxiv.org/abs/1706.03847)
- This code is based on [pyGRU4REC](https://github.com/yhs-968/pyGRU4REC) that is implemented by Younghun Song (yhs-968) and [original Theano code written by the authors of the GRU4REC paper](https://github.com/hidasib/GRU4Rec)
- This Version supports TOP1, BPR, TOP1-max, BPR-max, and Cross-Entropy Losses.

## Requirements
- PyTorch 0.4.1
- Python 3.5
- pandas
- numpy 1.14.5

## Usage

### Dataset
RecSys Challenge 2015 Dataset can be retreived from [HERE](https://2015.recsyschallenge.com/)

### Pre processing data
- You need to run preprocessing.py to obtain training data and testing data. In the paper, only the training set was used, the testing set is ignored.
- The training set itself is divided into training and testing where the testing split is the last day sessions.

The format of data is similar to that obtained from RecSys Challenge 2015:
- Filenames
    - Training set should be named as `recSys15TrainOnly.txt`
    - Test set should be named as `recSys15Valid.txt`
- Contents
    - `recSys15TrainOnly.txt`, `recSys15Valid.txt` should be the tsv files that stores the pandas dataframes that satisfy the following requirements:
        - The 1st column of the file should be the integer Session IDs with header name SessionID
        - The 2nd column of the file should be the integer Item IDs with header name ItemID
        - The 3rd column of the file should be the Timestamps with header name Time
        
### Training and Testing
The project have a structure as below:

```bash
├── GRU4REC-pytorch
│   ├── checkpoint
│   ├── data
│   │    ├── preprocessed_data
│   │    │    ├── recSys15TrainOnly.txt
│   │    │    ├── recSys15Valid.txt
│   │    ├── raw_data
│   │    │    ├── yoochoose-clicks.dat
│   ├── lib
│   ├── main.py
│   ├── preprocessing.py
│   ├── tool.py
```
`tool.py` can be used to get 1/8 last session from `yoochoose-clicks.dat`

In GRU4REC-pytorch

Training 
```bash
python main.py
```

Testing
```bash
python main.py --is_eval --load_model checkpoint/CHECKPOINT#/model_EPOCH#.pt
```
### List of Arguments accepted
```--hidden_size``` Number of Neurons per Layer (Default = 100) <br>
```--num_layers``` Number of Hidden Layers (Default = 1) <br>
```--batch_size``` Batch Size (Default = 50) <br>
```--dropout_input``` Dropout ratio at input (Default = 0) <br>
```--dropout_hidden``` Dropout at each hidden layer except the last one (Default = 0.5) <br>
```--n_epochs``` Number of epochs (Default = 10) <br>
```--k_eval``` Value of K used durig Recall@K and MRR@K Evaluation (Default = 20) <br>
```--optimizer_type``` Optimizer (Default = Adagrad) <br>
```--final_act``` Activation Function (Default = Tanh) <br>
```--lr``` Learning rate (Default = 0.01) <br>
```--weight_decay``` Weight decay (Default = 0) <br>
```--momentum``` Momentum Value (Default = 0)  <br>
```--eps``` Epsilon Value of Optimizer (Default = 1e-6)  <br>
```--loss_type``` Type of loss function TOP1 / BPR / TOP1-max / BPR-max / Cross-Entropy (Default: TOP1-max) <br>
```--time_sort``` In case items are not sorted by time stamp (Default = 0) <br>
```--model_name``` String of model name. <br>
```--save_dir```  String of folder to save the checkpoints and logs inside it (Default = /checkpoint).<br>
```--data_folder``` String of the directory to the folder containing the dataset. <br>
```--train_data```  Name of the training dataset file (Default = `recSys15TrainOnly.txt`)<br>
```--valid_data```  Name of the validation dataset file (Default = `recSys15Valid.txt`)<br>
```--is_eval``` Should be used in case of evaluation only using a checkpoint model. <br>
```--load_model``` String containing the checkpoint model to be used in evaluation. <br>
```--checkpoint_dir```  String containing directory of the checkpoints folder. <br>


## Results

Different loss functions and different parameters have been tried out and the results can be seen from [HERE](https://docs.google.com/spreadsheets/d/19z6zFEY6pC0msi3wOQLk_kJsvqF8xnGOJPUGhQ36-wI/edit#gid=0)
