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
| Proposal Presentation |                                         |  October 7th and 9th   |
| Proposal              | 2 pages                                 |  Friday, October 11th  |
| Progress Report       | 2 pages                                 |  Friday, November 8th  |
| Final Presentation    | In-class presentation                   |  December 2nd and 4th  |
| Final Report          | Minimum 5 pages, IEEE conference format | Tuesday, December 10th |

### Project Proposal

1. Project title and list of group members.
2. Overview of project idea. This should be approximately half a page long.
3. A short literature survey of 4 or more relevant papers. The literature review should
   take up approximately one page.
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
3. Methods: overview of your proposed method, Intuition on why should it be better than
   the state of the art, and details of models and algorithms that you developed.
4. Results: Description of your experiments and results and a list of questions your
   experiments are designed to answer.
5. Conclusion: discussion and future work.

****
### Data and Task
#### Project Goal  
The goal of this project is to integrate powerful deep neural networks and statistical methods with a Probabilistic Graphical Model (PGM) to address sequence labeling tasks on a heterogeneous medical care dataset.

---

##### Reference Papers

##### 1. Bidirectional LSTM-CRF Models for Sequence Tagging  
This paper represents one of the earliest successful applications of deep learning combined with CRF models for sequence labeling. We adopt the core idea from this work, where features and potential functions of a CRF model are generated using deep neural networks. While the authors use a Bi-LSTM architecture for feature extraction, we will leverage modern, state-of-the-art large neural network models like BERT to handle more complex datasets.

##### 2. Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials  
Most CRF models, aside from certain special cases like linear-chain CRFs, do not have efficient closed-form solutions. To address this, we will employ mean-field approximation for solving more general CRF models in the context of complex prediction tasks. By minimizing the KL divergence as the loss function, the mean-field approximation can be reduced to a fixed-point iteration, providing a faster and more efficient way to train and infer CRF models while maintaining appropriate accuracy.

##### 3. Conditional Random Fields as Recurrent Neural Networks  
This paper reformulates the mean-field approximation for fully connected CRFs using a CNN-RNN structure. For practical purposes, we can utilize advanced deep learning frameworks such as TensorFlow or PyTorch to accelerate the computation process, allowing us to efficiently solve complex CRF models in large-scale tasks.



#### Data Source
Here are some potential topics and data sources:
- Harmful Brain Activity Classification
  - Data description 
    - This data contains electroencephalography (EEG) signals from hospital patients collected by HMS.
    - The goal of analysing this dataset is to detect six interested patterns of harmful brain signal including: 
      seizure (SZ), generalized periodic discharges (GPD), 
      lateralized periodic discharges (LPD),
      lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA).
    - This dataset comes from a competition on Kaggle platform ended April 8, 2024. The data is still available on
      https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview
  - Project Proposal
    - The data contains raw signal spectrograms and processed signal time series, we plan to split the data in to
      segments according time steps, and labeling each segments with an interested pattern. 
      The segments splitting and feature extraction
      will be done with neural networks and the output CRF layer will be set as a fully connect CRF. Since we may need 
      to consider the heterogeneity of the data, thus some external variables may be introduced to the CRF layer, 
      this may need slight changes to the existing learning and inference methods of CRF.
- n2c2: National NLP Clinical Challenges
  - The dataset contains clinical notes from patients with various medical conditions and has annotations for named
    entities such as diseases, symptoms, treatments, and medications, which will allow for training and evaluating
    NER models.
  - The NER task is a standard sequence labeling task, thus we may not need structure alignment for exiting 
   state-of-the-art model.
  - The data is available after submitting a DUA for access on https://www.i2b2.org/NLP/DataSets/
