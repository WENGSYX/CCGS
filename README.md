# Cross-modal Contrastive Global-Span



## Updates

- 2022/10/12 updates codes 🏆

## Main Method

![main](./image/main.png)

###### We introduce a novel task, named video corpus visual answer localization (VCVAL), which aims to locate the visual answer in a large collection of untrimmed, unsegmented instructional videos using a natural language question. This task requires a range of skills - the interaction between vision and language, video retrieval, passage comprehension, and visual answer localization. To solve these, we propose a cross-modal contrastive global-span (CCGS) method for the VCVAL, jointly training the video corpus retrieval and visual answer localization tasks. More precisely, we enhance the video question-answer semantic by adding element-wise visual information into the pre-trained language model, and designing a novel global-span predictor through fusion information to locate the visual answer point. The Global-span contrastive learning is adopted to differentiate the span point in the positive and negative samples with the global-span matrix. We have reconstructed a new dataset named MedVidCQA and benchmarked the VCVAL task, where the proposed method achieves state-of-the-art (SOTA) both in the video corpus retrieval and visual answer localization tasks.

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

conda create -n CCGS python==3.7
conda activate CCGS
```

#### Installing Python Libraries

```plain
# preparing environment
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tqdm transformers sklearn pandas numpy glob accelerate sentencepiece
```

## MedVidCQA

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

## Cite
```
@article{li2022learning,
  title={Learning to Locate Visual Answer in Video Corpus Using Question},
  author={Li, Bin and Weng, Yixuan and Sun, Bin and Li, Shutao},
  journal={arXiv preprint arXiv:2210.05423},
  year={2022}
}
```
