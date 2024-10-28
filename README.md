# CSE 5830

+ Description: Probabilistic Graphical Models
+ Instructor: Sheida Nabavi
+ Dates: 2024/08/26 - 2024/12/06
+ Days: Monday, Wednesday
+ Times: 14:30 - 15:45
+ Location: ITE 127
+ Team:
  - Xiaohui Yin (xiaohui.yin@uconn.edu)
  - Shiying Xiao (shiying.xiao@uconn.edu)
  - Xiaohang Ma (xiaohang.ma@uconn.edu)

## Course Project

| Event                 | Detail                                  |          Due           |
|-----------------------|-----------------------------------------|:----------------------:|
| Proposal Presentation |                                         | October 7th and 9th    |
| Proposal              | 2 pages                                 | Friday, October 11th   |
| Progress Report       | 2 pages                                 | Friday, November 8th   |
| Final Presentation    | In-class presentation                   | December 2nd and 4th   |
| Final Report          | Minimum 5 pages, IEEE conference format | Tuesday, December 10th |

****
### Nvidia GPU Resource
Use the following account to get access to the Nvidia V100 GPU for training and inference, 
the project data was pre-downloaded from the HMS kaggle competition.
``````bash
URL: https://learn.nvidia.com/
Login Account: xhma572737202@gmail.com
Password: J@vN.L7)'ap3TPp
``````
To launch the jupyter notebook, access the Fundamentals of Deep Learning Course at 'My Learning' Page. 
Then Start the course and click Launch.
****
### Project Proposal

1. Project title and list of group members.
2. Overview of project idea. This should be approximately half a page long.
3. A short literature survey of 4 or more relevant papers. The literature review
   should take up approximately one page.
4. Description of potential data sets to use for the experiments.
5. Plan of activities, and how you plan to divide up the work.

### Project Progress Report

1. Project progress and accomplishments.
2. Preliminary results.
3. Challenges and problems (if any).
4. Changes to the original plan (if any).
5. Remaining work and plan to finish the project.
6. Anything we need to get updated about it.

### Final Report

1. Introduction: problem definition and motivation.
2. Background & Related Work: background info and literature survey.
3. Methods: overview of your proposed method, Intuition on why should it be
   better than the state of the art, and details of models and algorithms that
   you developed.
4. Results: Description of your experiments and results and a list of questions
   your experiments are designed to answer.
5. Conclusion: discussion and future work.

---

## Project Overview

This project aims to integrate powerful deep neural networks and statistical
methods with a probabilistic graphical model (PGM) to effectively tackle
sequence labeling tasks on a heterogeneous medical care dataset.

### References

1. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://doi.org/10.48550/arXiv.1508.01991)

This paper represents one of the earliest successful applications of deep
learning integrated with conditional random field (CRF) models for sequence
labeling. We build on the core idea of this work, where deep neural networks
are used to generate features and potential functions for a CRF model.
While the original authors employ a bidirectional LSTM (Bi-LSTM) architecture
for feature extraction, we will leverage modern, state-of-the-art large neural
network models, such as bidirectional encoder representations from transformers
(BERT), to tackle more complex datasets.

2. [Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://proceedings.neurips.cc/paper_files/paper/2011/file/beda24c1e1b46055dff2c39c98fd6fc1-Paper.pdf)

Most CRF models, except for specific cases like linear-chain CRFs, lack
efficient closed-form solutions. To overcome this, we will employ mean-field
approximation to solve more general CRF models in the context of complex
prediction tasks. By minimizing the Kullback-Leibler (KL) divergence as the loss
function, the mean-field approximation can be reduced to a fixed-point iteration,
providing a faster and more efficient approach for training and inference while
maintaining suitable accuracy.

3. [Conditional Random Fields as Recurrent Neural Networks](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Conditional_Random_Fields_ICCV_2015_paper.pdf)

This paper reformulates the mean-field approximation for fully connected CRFs
using a CRF-RNN (recurrent neural network) structure. For practical
implementation, we leverage advanced deep learning frameworks, such as
TensorFlow or PyTorch, to accelerate computation process, enabling efficient
solutions to complex CRF models in large-scale tasks.

4. [An end-to-end approach to segmentation in medical images with CNN and posterior-CRF](https://www.sciencedirect.com/science/article/pii/S136184152100356X?ref=pdf_download&fr=RR-2&rr=8d6e21ec8c5b3b8d)

This paper fulfilles the End-to-end CNN-CRF architecture for segmentation of CT image.
We will use the Posterior-CRF framework proposed by this paper, that is,  
the CNN will be integrated with a U-net structure and generating a feature maps 
with the same spatial scale as the original input of CNN. And the unary & binary features of CRF layer will be based on 
the feature maps.

[//]: # (We will integrate the U-net structure and training procedure from this paper with )

5. [A Multi-Stream Graph Convolutional
   Networks-Hidden Conditional Random Field
   Model for Skeleton-Based Action Recognition](https://ieeexplore.ieee.org/document/9000721)

This paper proposed a novel method to construct a Graph
Convolutional Networks-Hidden conditional Random Field model for human action recognition. 
The introduced HCRF will assign hidden labels and align their compatibilities for the features 
and the likelihood of the observed label 
could be obtained by summation over all possible assignments of hidden state.
That exactly meats the goal of our works, since in our data, we can only observe the label of the whole egg or 
spectrogram, but those pixels that are active for the label prediction is unknown. Thus, we will introduce 
a hidden CRF layer for the discrimination of which part of the image the model should pay attention to.
The major difference of our approach with this work is that, our hidden is fully connected, thus message passing and 
belief propagation are not trackable  in our case. For approximately obtaining the log-likelihood of our model
 a mean field variational inference will be used for the efficiency calculation of the loss function.
### Potential Topics and Data Sources

1. [Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)

+ Data Description:
  - This dataset contains electroencephalography (EEG) signals from hospital
    patients, collected by Harvard Medical School. It was originally part of a
    Kaggle competition that ended on April 8, 2024.
  - The goal of analyzing this dataset is to detect six patterns of harmful
    brain activity, including seizure (SZ), generalized periodic discharges (GPD),
    lateralized periodic discharges (LPD), lateralized rhythmic delta activity
    (LRDA), and generalized rhythmic delta activity (GRDA).
+ Project Proposal:
  - The data contains raw signal spectrograms and processed signal time series.
    We plan to split the data in to segments based on time steps, labeling each
    segments with the corresponding brain activity pattern. Segmenting and
    feature extraction will be handled using neural networks, with a fully
    connected CRF layer as the output. Given the potential heterogeneity of the
    data, we may need to introduce external variables into the CRF layer. This
    could require slight modifications to the existing learning and inference
    methods of CRF.

2. [n2c2: National NLP Clinical Challenges](https://www.i2b2.org/NLP/DataSets)

+ Data Description:
  - The dataset contains clinical notes from patients with various medical
    conditions. It includes annotations for named entities such as diseases,
    symptoms, treatments, and medications, which can be used to train and
    evaluate named entity recognition (NER) models. The data is available after
    submitting a Data Use Agreement (DUA) for access.
+ Project Proposal:
  - Given that NER is a standard sequence labeling task, structural alignment
    with existing state-of-the-art models might not be strictly necessary.

