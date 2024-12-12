# UNet-HCRF Integration for EEG Pattern Recognition

## Overview

This project introduces a framework that combines deep learning and
probabilistic graphical models to improve the accuracy of electroencephalography
(EEG) pattern recognition. By integrating UNet for feature extraction with a
hidden conditional random field (HCRF) model and employing mean-field
approximation for efficient inference, the approach offers a robust and
efficient solution for EEG analysis.

## Course

+ Description: CSE 5830 Probabilistic Graphical Models
+ Instructor: Sheida Nabavi
+ Dates: 2024/08/26 - 2024/12/06
+ Days: Monday, Wednesday
+ Times: 14:30 - 15:45
+ Location: ITE 127
+ Team:
  - Xiaohui Yin (xiaohui.yin@uconn.edu)
  - Shiying Xiao (shiying.xiao@uconn.edu)
  - Xiaohang Ma (xiaohang.ma@uconn.edu)

| Event                 | Detail                                  |          Due                                        |
|-----------------------|-----------------------------------------|:---------------------------------------------------:|
| Proposal Presentation |                                         | October 7th and 9th                                 |
| Proposal              | 2 pages                                 | Friday, October 11th                                |
| Progress Report       | 2 pages                                 | Friday, November 8th                                |
| Final Presentation    | In-class presentation                   | ~~December 2nd and 4th~~ December 9th               |
| Final Report          | Minimum 5 pages, IEEE conference format | ~~Tuesday, December 10th~~ Wednesday, December 11th |

---

## Dataset

[Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
contains EEG signals from hospital patients, collected by Harvard Medical School.
It was originally part of a Kaggle competition that ended on April 8, 2024.

The goal of analyzing this dataset is to detect six patterns of harmful brain
activity, including seizure (SZ), generalized periodic discharges (GPD),
lateralized periodic discharges (LPD), lateralized rhythmic delta activity
(LRDA), and generalized rhythmic delta activity (GRDA).

---

## Pretrain UNet Model

Please refer to [EfficientNetB0 Starter - [LB 0.43]](https://www.kaggle.com/code/cdeotte/efficientnetb0-starter-lb-0-43#Train-Scheduler)
for training a UNet model 

