# Documentation of the types of risk and possible mitigations associated with different types of machine learning models

Initial Authors: Jim Smith, Alba Crespi Boixader 2025
### V0.1: April 2025

<div style="height:10px;background:black;width:400"></div>

## Description
The document aims to capture the types of privacy risk and mitigations associated with different forms of Machine Learning.

It is a counter-part to the [statbarns taxonomy](https://doi.org/10.1007%2F978-3-031-69651-0_19) for 'traditional analytic outputs from TREs.

The first group below is a special case due to the way their algorithms work.

The subsequent groups are  categorised according to the type of output that they produce.  
Specifically the key question is whether models are designed to predict:
- numerical values (*regression models*),
- class probabilities (*classification*),
- semi-structured outputs (e.g. segmented images) or
- unstructured outputs (e.g free text responses).
- a pre-processed *embedding* of the input data.
   These sometimes will be described using phrases like *Foundation Models* or *encoder-decoder* architectures models.
  They are typically designed so that they can subsequently be rapidly repurposed for a range of related classfication or regression tasks.

It is important to note that almost all Machine Learning Algorithms can be trained to create classification or regression models.  
Deep Learning models are typically needed for the last three types of output.  




<div style="height:10px;background:black;width:400"></div>

## Types of risk considered

- Model *explicitly* encodes data
- Small-group reporting ( which can enable Re-identification / Attribute Inference)
- Class Disclosure
- Membership Inference
- Attribute Inference for known members
- Model Inversion
- Model can be triggered to regurgitate *implicitly* stored training data

<div style="height:10px;background:black;width:400"></div>

## Group 1: Instance-Based Learning.

### Examples of Instance-Based Learning
- Support Vector Machines (SVMs) for example Support Vector Classifiers and Support Vector Regressors.
- Radial Basis Function Networks
- k-Nearest Neighbours
- Case-based reasoning
- kernel models- alternative name given to a broad class which includes SVMs
- Self Organising Map (SOM),
- Learning Vector Quantization (LVQ),
- Locally Weighted Learning (LWL),
  
### Principle Risk: Model explicitly contains training data 
The risk from this group of models occurs because they directly embed members of the training data in the model.

- This  risk applies **regardless** of the type of output.

### Mitigation
The mitigation that **must** be applied is that the preprocessing sufficiently removes any personal identifiers.

   - It could be argued this effectively means that the TRE would be comfortable with releasing the pre-processed data.
   - For example, the data might be transformed into a synthetic dataset via a 'Differentially' Private' embedding.

-  This mitigation does not necessarily guard against other risks, such as class disclosure.

### Secondary Risks:

All  risks below apply and should also be checked for - although the primary mitigation *may* be sufficient for these also.
- Small Group Reporting
- Class Disclosure
- Attribute Inference
  

<div style="height:10px;background:black;width:400"></div>

## Group 2: Regression Models
These are models that have been trained to make a numerical prediction - in which we do not include probability of some event or class occurring.  
Examples from different domains include: air pollutant levels, risk of re-offending, duration of hospital stay, etc.

In general the risks for this group are the same as for well understood Linear/Logistic/Logit Regression.

### Examples of Regression Models
Regression models can be created with most Machine Learning Algorithms, as well as statistical techniques such as the *ARIMA* family of models.   



### Principal risk: *Small Group Reporting*:
- the model should not be specified so completely that  nay partof it described a small group of records
- typically this means stipulating a lower limit on the *residual degress of freedom*
   number_of_records - number_of_trainable_parameters_in_the_model

Some models may implicitly or explicitly perform *piece-wise regression* in  which case each sub-group should be checked for size.

- i.e., are there some output values which are only predicted for a small number of training records

### Secondary risks: 
- Class disclosure
    - but this probably only occurs when a models is trained to predict levels of more 2 or more variables.

### Mitigations for Regression Models
1. Models pass  'Structural Attacks'
- for classification models these  check for residual degrees of freedom, class disclosure and k-anonymity 
- a  small amount of work is needed  to adapt to regression models.
- prioritisaion to be decided by the community
- These are relatively cheap to run as they do not involve training any new models.

2. Model Query Controls
- might be appropriate for extremely large regression models with multiple predicted variables

<div style="height:10px;background:black;width:400"></div>

## Group 3: Classification Models
These models  are designed to predict the probability that a record is associated with different output classes.
This could be a single value *P(voting in next election)* or the likelihoods associated with a finite set of classes e.g. *P(votes for party X)* or linking genetic/health records to different disease diagnoses.

### Examples of Classification Models
Classification models can be created with most Machine Learning Algorithms

### Risks  and Mitigations
1. Small-group reporting ( which can enable Re-identification / Attribute Inference)
2. Class Disclosure
3. Membership Inference
4. Attribute Inference for known members
5. Model Inversion

<div style="height:10px;background:black;width:400"></div>

## Models producing semi-structured outputs

### Examples
- Vision-based  models that autoamtically segment "regions of interest" in an image.
  
### Risks  and Mitigations
1. Small-group reporting ( which can enable Re-identification / Attribute Inference)
2. Class Disclosure
3. Membership Inference
4. Attribute Inference for known members
5. Model Inversion

<div style="height:10px;background:black;width:400"></div>

## Models producing unstructured outputs (e.g. Natural Language).

### Examples
- Models that  produce summaries of inputs (could be text or images)
- Chatbots
- **Foresight**
- 
### Risks  and Mitigations
1. Model can be triggered to regurgitate implictly stored training data
2. Small-group reporting ( which can enable Re-identification / Attribute Inference)
3. Class Disclosure
4. Membership Inference
5. Attribute Inference for known members
6. Model Inversion

Mitigation 1:  *alignment* via human-in-the-loop-reinforcmeent-learning, 
- used for commercial Large Language Model
- but recent reportde[these can be broken](https://www.theguardian.com/technology/2024/apr/03/many-shot-jailbreaking-ai-artificial-intelligence-safety-features-bypass?CMP=Share_iOSApp_Other)

<div style="height:10px;background:black;width:400"></div>

## Foundation models
