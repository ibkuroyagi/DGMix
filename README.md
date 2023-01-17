# DG-Mix: Domain Generalization for Anomalous Sound Detection Based on Self-Supervised Learning

This repository is a non official reproduction of "DG-Mix: Domain Generalization for Anomalous Sound Detection Based on Self-Supervised Learning."

Details of the method are written [here](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Nejjar_31.pdf).  
This paper is published at DCASE2022 Workshop.

## Requirements

- Python 3.9+
- Cuda 11.3

## Setup

```bash
git clone https://github.com/ibkuroyagi/DGMix.git
cd DGMix/tools
make
```

## Recipe

- [dcase2022-task2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring): The main challenge of this task is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data.

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd scripts

$ ./job.sh

```

## Results

Results of this model can be seen in scripts/exp/all/dgmix.original/checkpoint-120epochs/score.md.  
These scores are harmonic mean of AUC [%].  

|             |     AUC |   bearing_auc |   fan_auc |   gearbox_auc |   valve_auc |   slider_auc |   ToyCar_auc |   ToyTrain_auc |
|:------------|--------:|--------------:|----------:|--------------:|------------:|-------------:|-------------:|---------------:|
| source_dev  | 55.6417 |       52.3611 |   54.3536 |       69.2407 |     49.9643 |      66.6902 |      57.2063 |        46.7799 |
| source_eval | 55.0961 |       55.8166 |   51.8338 |       58.1005 |     53.6397 |      63.544  |      53.0038 |        51.5879 |
| target_dev  | 51.5993 |       55.1491 |   55.9276 |       65.069  |     36.9344 |      52.9252 |      50.6347 |        53.8774 |
| target_eval | 52.6867 |       54.7691 |   46.7551 |       53.7797 |     55.2403 |      51.855  |      54.2659 |        53.1774 |

## Reference

```
@inproceedings{Nejjar2022,
    author = "Nejjar, Ismail and Meunier-Pion, Jean and Frusque, Gaetan and Fink, Olga",
    title = "DG-Mix: Domain Generalization for Anomalous Sound Detection Based on Self-Supervised Learning",
    booktitle = "Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022)",
    address = "Nancy, France",
    month = "November",
    year = "2022",
}
```

## Author of this repository

Ibuki Kuroyanagi ([@ibkuroyagi](https://github.com/ibkuroyagi))  
E-mail: `kuroyanagi.ibuki<at>g.sp.m.is.nagoya-u.ac.jp`
