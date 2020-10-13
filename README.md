# TDA-ReCTS: A Validation Set for Text Detection Ambiguity

## Introduction
TDA-ReCTS is a validation set for benchmarking text detection ambiguity,
which contains 1,000 ambiguous images selected from the training set of [ICDAR 2019 ReCTS](https://rrc.cvc.uab.es/?ch=12).

This repository includes TDA-ReCTS's training list, validation list, and evaluation script.
The code of AE TextSpotter is released at https://github.com/whai362/AE_TextSpotter.

<div align="center">
  <img src="https://github.com/whai362/TDA-ReCTS/blob/master/images/examples.png" width="600">
</div>
<p align="center">
  Fig. 1: Some exmaples in validation list.
</p>


## Requirements
* Python3
* mmcv==0.2.13
* Polygon3==3.0.8
* editdistance

## Generation of Training and Validation List
```shell script
python gen_train_val_list.py --data_root ${RECTS_ROOT}
```

The root of ReCTS should be: 
```
RECTS_ROOT
├── train
│   ├── img
│   ├── gt
├── test
│   ├── img
```

## Submission Format
A json file that includes the prediction of all images. Here is an [example submission](https://github.com/whai362/TDA-ReCTS/blob/master/eval_script/example_pred.json).
```
[{"img_name": "train_ReCTS_001213.jpg", 
  "points": [[[x_00, y_00], [x_01, y_01],..., [x_0n, y_0n]], 
             [[x_11, y_11], [x_12, y_11],..., [x_1n, y_1n]],
             ...,
             [[x_m1, y_m1], [x_m2, y_m1],..., [x_mn, y_mn]]], 
  "scores": [score_0, score_1, ..., score_m], 
  "texts": [text_0, text_1, ..., text_m]}
]
```
## Evaluation Script
```shell script
python eval_script/eval.py --gt eval_script/val_gt.zip --pred eval_script/example_pred.json
```
This evaluation script based on that of [BDD100K](https://github.com/ucbdrive/bdd100k/blob/master/bdd100k/evaluate.py).
The output of the evaluation script should be:
```
[Best F-Measure] p: 84.79, r: 78.16, f: 81.34, 1-ned: 51.08, best_score_th: 0.569
[Best 1-NED]     p: 86.55, r: 76.32, f: 81.11, 1-ned: 51.60, best_score_th: 0.626
```
Here, 1-NED is a metric for text recognition, which can be formulated as:
<div align="center">
  <img src="https://github.com/whai362/TDA-ReCTS/blob/master/images/1_ned.gif", width="300">
</div>
<p align="center">
  Eqn. 1: $N_{FP}$, $N_{TP}$ and $N_{GT}$ mean the number of false-positive, true-positive and ground-truth text lines, respectively. $S_i$ and $\hat{S}_i$ represent predicted transcription and ground-truth transcription, respectively.
</p> 

## Citation
If you use this work in your research, please cite us.
```
@inproceedings{wenhai2020ae,
  title={AE TextSpotter: Learning Visual and Linguistic Representation for Ambiguous Text Spotting},
  author={Wang, Wenhai and Liu, Xuebo and Ji, Xiaozhong and Xie, Enze and Liang, Ding and Yang, ZhiBo and Lu, Tong and Shen, Chunhua and Luo, Ping},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
