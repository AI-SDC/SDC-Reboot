# Documentation of the types of risk and possible mitigations associated with different types of machine learning models

Initial Authors: Jim Smith, Alba Crespi Boixader 2025
### V0.1: April 2025

<div style="height:10px;background:black;width:400"></div>

## $${\text{\color{blue}Purpose \ of\ document}}$$
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

## $${\text{\color{blue}Types\ of\ risk\ considered}}$$

- Model *explicitly* encodes data\
  *brief description*
- Small-group reporting ( which can enable Re-identification / Attribute Inference)\
  *brief description*
- Class Disclosure\
  *brief description*
- Membership Inference\
  *brief description*
- Attribute Inference for known members\
  *brief description*
- Model Inversion\
  *brief description*
- Model can be triggered to regurgitate *implicitly* stored training data\
  *brief description*

<div style="height:10px;background:black;width:400"></div>

## $${\text{\color{blue}Group\ 1:\ Instance-Based\ Models.}}$$
These models are created by a group of algorithms that make predictions based on distances to explicitly included training records.
The best knowm example is 1-Nearest Neighbour which effectively says *"What's the closest thing I've seen already?"*
### Examples of Instance-Based Models
- Support Vector Machines (SVMs) for example Support Vector Classifiers and Support Vector Regressors.
- Radial Basis Function Networks.
- k-Nearest Neighbours.
- Case-based reasoning.
- kernel models- alternative name given to a broad class which includes SVMs.
- Self Organising Map (SOM).
- Learning Vector Quantization (LVQ).
- Locally Weighted Learning (LWL).
  
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

## $${\text{\color{blue}Group\ 2:\ Regression\ Models}}$$
These are models that have been trained to make a numerical prediction - in which we do not include probability of some event or class occurring.  
Examples from different domains include: air pollutant levels, risk of re-offending, duration of hospital stay, etc.

In general the risks for this group are the same as for well understood Linear/Logistic/Logit Regression.

### Examples of Regression Models
Regression models can be created with most Machine Learning Algorithms, as well as many different statistical techniques such as the *ARIMA* family of models.   



### Principal risk: *Small Group Reporting*:
- the model should not be specified so completely that  any part of it describes a small group of records
- typically this means stipulating a lower limit on the *residual degrees of freedom* :  
  ```DoF =  number_of_records - number_of_trainable_parameters_in_the_model```

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

## $${\text{\color{blue}Group\ 3:\ Classification\ Models}}$$
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

## $${\text{\color{blue}Group\ 4:\ Models\ producing\ semi-structured\ outputs}}$$
*brief description to follow*
### Examples
- Vision-based  models that automatically segment "regions of interest" in an image.
  
### Risks  and Mitigations
1. Small-group reporting ( which can enable Re-identification / Attribute Inference)
2. Class Disclosure
3. Membership Inference
4. Attribute Inference for known members
5. Model Inversion

<div style="height:10px;background:black;width:400"></div>

## $${\text{\color{blue}Group5:\ Models\ producing\ unstructured\ outputs (e.g.\ Natural\ Language).}}$$
*brief description to follow*
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

Mitigation 1:  *alignment* via human-in-the-loop-reinforcement-learning, 
- used for commercial Large Language Models to try and prevent them giving certain responses
- but recent reports suggest that [these defences can be broken](https://www.theguardian.com/technology/2024/apr/03/many-shot-jailbreaking-ai-artificial-intelligence-safety-features-bypass?CMP=Share_iOSApp_Other)

<div style="height:10px;background:black;width:400"></div>

## $${\text{\color{blue}Group6:\ Foundation\ models}}$$

### TODO merge below - see note

  These type of models are pre-trained on vast amounts of general data, and then are fine-tuned, adapted or carefully engineered for specific applications. They are adaptable and proving, in many cases, more efficient than a single model for the use-case.

  With only a few foundation models, trained on a very limited number of datasets, the applications are vast. However, it poses a monopolistic power structure problem.

  They are based on deep neural networks and transfer learning. Which are well stablish method in the AI world. The main difference lie on being trained on extreme large amounts of data. Therefore the ability to be transferred across domains.

### Examples
Some examples of foundational models are:
- Open AI's GPT-Series
- BERT
- CLIP
- OpenLLaMa

The application specific fine-tuned models are mostly in the field of natural language processing (NLP), and well known examples include:
- ChatGPT
- DeepSeek
- Grok
- DALL-E

### Risks and mitigations

1. It is hard to understand and explain the behaviour.
2. Uncertainty on the capabilities.
3. Unclear the data which had been pre-trained on. Including, dataset bias, copyright and license.

Mitigation 1: Be cautious and aware of potential "unsettling" behaviour.

Mitigation 2: Foundation models can be *homogenised* by a few foundational models, train on a few datasets and/or a few organisations. However, the elevated costs of training such models means in practice, most people or organisations can't do it, exacerbating the monopoly of a few companies which own the large datasets.

Mitigation 3: Without a clear definition of the characteristics of the data it is hard to know which biases there might be. This can be fine-tuned by observations. 
For TREs or equivalent and developers, need to make sure they can host the pre trained model and data, and ensure which limitations, if any, there are for the derived model.

### Below written by JS while on train and nees ot be merged into the above
Broadly speaking these are a class of models that are trained on lots of (often unlabelled, almost always sequence-structured) data from a given domain. The goal is learn a useful `embedding' (effectively a preprocessing that maps high-dimensional data onto  to a smaller set of features) that captures all of the important information within a sequence and can be easily adapted to specific tasks. This embedding forms the bridge between the *encoder* (learned preprocessing) and the *decoder* which transforms the encoded inputs back into a form suitable for a task.

### Examples:
- Large Langiuage Models (Chat-GPT, Gemini etc.)
- Vision models (8DALL-E, Stable Diffusion etc.)
### Considerations  at project inception
Typically foundation models   are `pre-trained'  using a decoder that attempts to solve a task that does not require manually labelled data, and so can use huge datasets from the internet e.g.:
- predict the next token in a sequence for *large language models* (Chat-GPT etc.)
- removing randomly added 'noise' from an image   for *vision-based transformers* (Dall-E etc.)

Subsequently they can be retrained for a specific tasks using smaller amounts of unlabelled data.
Often this is done by only removing the 'head" (the final layer(s) of the *decoder*)  and replacing it with one suited to the new task (e.g. *N* output nodes for a classification problem with *N* different labels)

When a project wants to use a foundation model it is vital from project inception to  be clear whether researchers want to:
1. Import a pre-trained model, then put a new  `head' on it and train that for a new project **without affecting the rest of the foundation model**.
   - In this case they may not even need to export most of the model- just the prediction head. Therefore the 'head' can be treated as a simple regression or classification model (Groups 2 or 3).
   -  It would be reasonable to ask the researcher to separate their code into callable functions that do preprocessing (i.e. the unchanged 'body' of the foundation model) and prediction ('head'). This would allow  the TRE can run the various mitigation tests described above.
   -  If the head is simply a single output layer, then this is effectively  a linear or logistic regression model and can be evaluated as such.
2.  Import a pre-trained model, put a new head on it, then adapt the head **and the 'body' of foundation model weights**.
   - In this case potentially both the head and the body could be memorising trainign data and so need assessing
4.  Train a foundation model from scratch and then export it.

### Risks
In both these last two cases the full model needs to be evaluated for risks including:
- Model *explicitly* encodes data
- Small-group reporting ( which can enable Re-identification / Attribute Inference)
- Class Disclosure
- Attribute Inference for known members.
-  Membership Inference.

The first four risks  can be usually be measured without retraining the model, which may be plausible.

However, current tests to assess the risk of Membership Inference involve training several 'shadow' models of the same complexity, with slightly different datasets.
Whether this is plausible will depend entirely on the run-time it took to train the resarcher's model:
- it  **may** be possible for scenario 2, depending on the size of the training data and however many `epochs' (iterations) of training were used.
- is **highly unlikely** to be feasible for scenario 3 (training a foundation model from scratch).

### Mitigations:
- Senario 1: As per groups 2 and 3.
- Case 2: **possibly using a `Differentially Private Optimiser` for retraining.
  In this case it would be necessary to justify whether Class Disclosure is not relevant to the dataset.
- Case 2 and 3: Arguments that claim the fixed or learned preprocessing effrectively k-anonymises the data.

### References
Schneider, J., Meske, C. & Kuss, P. Foundation Models. Bus Inf Syst Eng 66, 221–231 (2024). https://doi.org/10.1007/s12599-024-00851-0