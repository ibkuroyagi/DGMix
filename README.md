# Anomalous Sound Detection with Pytorch

This repository is a non official reproduction of "DG-Mix: Domain Generalization for Anomalous Sound Detection Based on Self-Supervised Learning."

Details of the method are written [here](https://dcase.community/documents/workshop2023/proceedings/DCASE2023Workshop_Nejjar_31.pdf).  
This paper is published at DCASE2023 Workshop.

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

- [dcase2023-task2](https://dcase.community/challenge2023/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring): The main challenge of this task is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data.

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd scripts

# Run the recipe from scratch
$ ./job.sh

# You can change config via command line
$ ./job.sh --no <the_number_of_your_customized_yaml_config>

# You can select the stage to start and stop
$ ./job.sh --stage 1 --start_stage 3

# After all machine types have completed Stage 5, starting Stage 2.
# You can see the results at exp/all/**/score*.csv
$ ./job.sh --stage 2


```

## Citation

```
@inproceedings{Nejjar2023,
    author = "Nejjar, Ismail and Meunier-Pion, Jean and Frusque, Gaetan and Fink, Olga",
    title = "DG-Mix: Domain Generalization for Anomalous Sound Detection Based on Self-Supervised Learning",
    booktitle = "Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2023 Workshop (DCASE2023)",
    address = "Nancy, France",
    month = "November",
    year = "2023",
}
```

## Author of this repository

Ibuki Kuroyanagi ([@ibkuroyagi](https://github.com/ibkuroyagi))  
E-mail: `kuroyanagi.ibuki<at>g.sp.m.is.nagoya-u.ac.jp`
