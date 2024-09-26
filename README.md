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
In this project, we will combine the powerful deep neural networks and statistical methods with 
Probabilistic Graphical Model(PGM) to solve sequence labeling task on heterogeneous medical care dataset.

#### Reference Paper
- Bidirectional LSTM-CRF Models for Sequence Tagging:
  - This paper belongs to one of the earliest successful attempts on deep learning based CRF for sequence labeling. We 
    bring the core idea from this work that generating features and potential functions of CRF model from 
    deep neural networks. In this paper, the authors choose Bi-LSTM for feature extraction, however, we will
    apply modern state-of-the-art large neural network models like BERT on more complicated datasets.
  
- Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials:
  - Since most CRF model does not meet an efficient closed form solutions, except some special case like
    linear chain CRF. Thus, we will use the mean-field approximation for solving more general CRF models associated with
    more complex prediction task. By setting the KL divergence as the loss function, the mean-field approximation will 
    be reduced as a fixed point iteration, which leads to a faster way to learn and inference CRF models with a proper 
    accuracy.

- Conditional Random Fields as Recurrent Neural Networks
  - In this paper, the mean field approximation of fully connected CRF was reformulated by a CNN-RNN structure. Thus, 
    for practical computation task, we can use advanced toolbox for deep learning such as tensorflow or pytorch to 
    accelerate our computing procedure.


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
