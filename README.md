# Global-Span



## Updates

- 2022/10/12 updates codes



## Prerequisites

- python 3.7 with pytorch (`1.10.0`), transformers(`4.15.0`), tqdm, accelerate, pandas, numpy, glob, sentencepiece
- cuda10/cuda11

#### Installing the GPU driver

```shell script
# preparing environment
sudo apt-get install gcc
sudo apt-get install make
wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
sudo sh cuda_11.5.1_495.29.05_linux.run
```

#### Installing Conda and Python

```shell script
# preparing environment
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh

conda create -n VPTSL python==3.7
conda activate VPTSL
```

#### Installing Python Libraries

```plain
# preparing environment
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tqdm transformers sklearn pandas numpy glob accelerate sentencepiece
```

### Download Data

##### Download the MedVidQA dataset from [here ](mailto:wengsyx@gmail.com?subject=Get%20MedVidCQA%20Dataset)

) and place it in `./data` directory.



## Quick Start

### Get Best

```shell script
bash run.sh
```

> All our hyperparameters are saved to  `run.sh` file, you can easily reproduce our best results.

### Try it yourself



```shell script
python main.py --device 0 \
	--seed 42 \
	--maxlen 1300 \
	--epochs 30 \
	--batchsize 4 \
	--lr 1e-5 \
	--weight_decay 0

```

> In this phase, training and testing will be carried out.
>
> In addition, after each round of training, it will be tested in the valid and test sets. In our paper, we report the model with the highest valid set and its score in the test set





### Files

```shell script
-- data
-- log

-- main.py
-- model.py
-- utils.py

```
