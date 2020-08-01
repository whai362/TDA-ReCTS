# TDA-ReCTS: A Validation Set for Text Detection Ambiguity

## Introduction
TDA-ReCTS is a validation set for benchmarking text detection ambiguity,
which contains 1,000 ambiguous images selected from the training set of [IC19-
ReCTS](https://rrc.cvc.uab.es/?ch=12).

This repository includes TDA-ReCTS's training list, validation list, and evaluation script.

<div align="center">
  <img src="https://github.com/whai362/TDA-ReCTS/blob/master/images/examples.png">
</div>
<p align="center">
  Figure 1: Some exmaples in validation list.
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
A json file that includes the prediction of all images (an example "eval_script/example_pred.json")
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