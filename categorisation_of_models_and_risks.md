# Documentation of the types of risk and possible mitigations associated with different types of machine learning models

### Initial Authors: Jim Smith, Alba Crespi-Boixander 2025
### V0.1: April 2025

The document aims to capture the types of privacy risk and mitigations assocuayed with different forms of Machine Learning.

Generally speaking we will categorise them according to the type of output that they produce, i.e. whether it is:
- numerical values (*regression models*),
- class probabilities (*classification*),
- semi-structured outputs (e.g. segmented images) or
- unstructured outputs (e.g free text responses).
- a pre-processed *embedding* of the input data.
   These sometimes will be described using phrases like *Foundation Models* or *encoder-decoder* architectures models.
  They are typically designed so that they can subsequently be rapidly repurposed for a range of related classfication or regression tasks.

It is important to note that almost all Machine Learning Algorithms can be trained to create classfication or regression models.  
Deep Learning models are typically needed for the last three types of output.  





## Types of risk

- Model explicitly encodes data
- Small-group reporting ( which can enable Re-identification / Attribute Inference)
- Class Disclosure
- Membership Inference
- Attribute Inference for known members
- Model Inversion

## Group 1: Models that *explicitly* embed training data.

- The risk from this group of models occurs because they directly embed members of the training data in the model.

- This  risk applies **regardless** of the type of output.

- The mitigation that **must** be applied is that the preprocessing sufficiently removes any personal identifiers.

   - It could be argued this effectively means that the TRE would be comfortable with releasing the pre-processed data.
   - For example, the data might be transformed into a synthetic dataset via a 'Differentially' Private' embedding.

-  This mitigation does not necessarily guard against other risks, such as class disclosure.

### Examples of group 1
- Support Vector Machines (SVMs) for example Support Vector Classifiers and Support Vector Regressors.
- Radial Basis Function Networks
- k-Nearest Neighbours
- Case-based reasoning
- kernel models (broard class whixgh inlcudes SVMs)


## Group 2: Regression Models
These are models that have been trained to make a numerical prediction - in which we do not include probability of some event or class occurring.  
Examples from different domains include: air pollutant levels, risk of re-offending, duration of hospital stay, etc.

*Regression models can be created with most Machine Learning Algorithms*.   

In general the risks are the same as for well understood Linear/Logistic/Logit Regression.

The main risk is *Small Group Reporting*:
- the model should not be specified so completely that  nay partof it described a small group of records
- typically this means stipulating a lower limit on the *residual degress of freedom*
   number_of_records - number_of_trainable_parameters_in_the_model

Some models may implicitly or explicitly perform *piece-wise regression* in  which case each sub-group should be checked for size.

- i.e., are there some output values which are only predicted for a small number of training records

A secondary risk might be **Class disclosure** 
- but this probably only occurs when a models is trained to predict levels of more 2 or more variables.



## Classification Models




## Models producing semi-structured outputs


## Models producing unstructured outputs (e.g. Natural Language).

## Foundation models
