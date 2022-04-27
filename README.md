# household_speaker_recognition

**WORK IN PROGRESS**

Please contact @sholokovalexey and @underdogliu if having any question.

This is the associated baseline system for our work on Speaker Odyssey, focusing on household speaker recognition. 

Code in this repo is subjected to baseline experiments with limited number of protocols. Therefore, it needs re-factoring and incremental updates as the research proceeds.

## Usage

### Pre-requisites
Python3.8+. We tested our code on python 3.8 and 3.9.

Run `pip install -r requirements.txt` to config the environment. For python virtual environment, please check related instructions in [Virtualenv](https://virtualenv.pypa.io/en/latest/) or [Conda](https://docs.anaconda.com/anaconda/user-guide/getting-started/).

### Runner
1. Download the speaker embeddings from [this link](https://drive.google.com/drive/folders/1eEC0IOV2KJR-7TV2v56o66A5_lazTCnV?usp=sharing) and store them in `${YOUR_PATH}/embeddings` (we will employ git LFS later).
2. Config the path of embeddings in `config.yaml` to `${YOUR_PATH}/embeddings`.
3. Run `scripts/run_all.sh` for empirical experiments across all baseline configurations, including active and passive enrollments. There are multiple other scripts for individual experiments. You can have a check on the scripts and related config files in `./configs` for more.


## Features

### Backend Algorithms
Whether we go for active or passive enrollment approach, we include the following recognizing algorithms:
* K-means clustering
* Variational Bayesian (VB) clustering
* Label propagation
* Aggelomerative hierarchical clustering (AHC)

For details about the backend algorithms we used, please read our paper.

### Scoring
We perform threshold centroid-based scoring with a fixed threshold.

### Dataset
We perform training and evaluation on two datasets:
* ASVspoof 2019, physical access (PA)
* VoxCeleb1


## Extension
For interested users who want to extend the toolkit and test new algorithms, please have a check on:
* `models.py` - for speaker recognition and scoring backend
* `clustering*.py` - for various clustering algorithms applied


## Citation
If you would like to use this repo, please cite our work:

```
@article{alexeyhousehold2022,
  title={Baselines and Protocols for Household Speaker Recognition},
  author={Alexey Sholokhov, Xuechen Liu, Md Sahidullah and Tomi Kinnunen},
  journal={Proc. Speaker Odyssey},
  year={2022}
}
```