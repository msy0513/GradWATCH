# GradWATCH-mian
Tracing Your Account: A Gradient-Aware Dynamic Window Graph Framework for Ethereum under Privacy-Preserving Services

This is a Python implementation of GradWATCH, as described in the following:
> Tracing Your Account: A Gradient-Aware Dynamic Window Graph Framework for Ethereum under Privacy-Preserving Services

## Requirements
For software configuration, all model are implemented in
- Python 3.9
- Torch 2.1.1
- torch-scatter 2.1.2+pt21cu121
- DGL 2.2.1+cu121
- CUDA 12.1
- scikit-learn 1.2.0
- numpy 1.26.3
- tqdm 4.64.1

## Data
The original transaction can be downloaded from the blockchain browser [page](https://goto.etherscan.com/txs?a=0x602809252600121dc4b9c0904148d07e4d5db26f).
Given the large size of the data files, we provide download links for access. For batch downloads, we recommend using the API interface provided by the hosting website. If you prefer to use our preprocessed data directly, please refer to the files in the dataset folder.


## Usage
Execute the following bash commands in the same directory where the code resides:
1. Process the dataset, generate MixTAG, and output the data to meet the input requirements of the model, which will be placed in the '/tornado-rule' or '/ens' folder:
  ```bash
 $ python prep_tc.py 
 $ python prep_ens.py 
  ```
2. Transaction-to-Account Mapping：
  ```bash
$ cd nodedemo
$ python input_embedding4.py 
  ```
Here, the initial embeddings of the nodes are generated based on the original transactions, and a reconstruction function is used to ensure that the newly generated embeddings have a statistical similarity with the original data.

3. Model training and testing：
```bash
$ python main.py
```
