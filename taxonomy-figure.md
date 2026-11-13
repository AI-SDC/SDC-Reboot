# Extending the statbarn to cover AI models trained on sensitive data

## TO-DOs
- add mitigation of restricting model capacity (via hyper-params or safe model wrappers)
- add operational measures

## Purpose and Core Concepts
This document aims to capture aspects of the disclsoiure control process for AI models trained on sensitive data.

It uses the concepts and relationships from the "Statbarns" framework as formalised in the statbarnsdc ontology [w3id.org/statbarnsdc](w3id.org/statbarnsdc)

Namely that types of output or models cabn be grouped into families - statbarns- which share vulnerability common sets of risks. 
- Each **statbarn** will typically be associated with a specific, different  combination of risks.
- Each **risk** is associated with:
    -  one or more checks
    - one or more potential mitigations.
- Each **check** has associated **evidence** and a threshold (_organisationalMeasure_)
- A **mitigation** might:
    - have its own _organisationalMeasure_ (e.g. rounding base, $\epsilon$ value for Differential Privacy) 
    - mitigate several risks.
    - reduce the **likelihood** or the **impact** of that risk occurring.


## AI-Statbarns
---

| Name | shorthand | Description | Examples | uri |
| ---| --- | --- | --- | --- | 
| Instance Based Models | `Category A'| Models that make predictions <br>via comparison to stored data | _K-Nearest Neighbours_, <br> _Support Vector Machines_ | tbc| 
| Sequence Based Models | `Category B'| Models that predict based on a sequence <br>of previous inputs and predictions |_Generative AI_ <br> _Recurrent Neural Networks_ |tbc| 
| Classification Models |`Category C1'|Models that predict one (or more) categorical label(s) for a record <br> independently of any other predictions they make | _DecisionTreeClassifier_,<br>_XGBoostClassifier_,<br>_MLPClassifier_ | tbc|
| Regression Models |`Category C2'|Models that predict one (or more) numerical value(s) for a record <br> independently of any other predictions they make | _DecisionTreeRegressor_,<br>_XGBoostRegressor_,<br>_MLPRegressor_ | tbc|
|Semi-Structured predictors|`Category C3'|Models that predict one (or more) numerical value(s) for a record <br> independently of any other predictions they make | _YOLO Object detection/segmentation_<br> | tbc|


## AI Risks
|Name | Shorthand | Description | uri |
|---|---|---|---|
|Extraction of<br> explicitly <br>stored training data |explicit| Model directly embeds traing data <br> that can be trivially extractecd| tbc|
|Extraction of <br>implicitly <br> stored data| implicit|Some inputss may "trigger" model to "regurgutate" training records|tbc|
|Class Disclosure| class disclosure | standard definition <br>can manifest as model partitioning <br> data space so that all records in a group<br> have zero probability of certain labels| https://w3id.org/statbarnsdc#ClassDisclosure|
|Small group reporting |small groups | extension of risk from tables where cells haveve low counts |https://w3id.org/statbarnsdc#LowCount|
|Low Residual Degrees of Freedom |LowDoF | The risk that a model with high numbers of parameters <br> relative to size of training data<br> may simply memorise records |https://w3id.org/statbarnsdc#LowDOF|
|Membership Inference| MIA |The risk thgat an attacker <br> armed with a complete record and the trained model,<br> can infer that the record was part of the training set | tbc|
|Attribute Inference |AIA| The risk that an attacker <br> armed with a partial record and a trained model <br> can infer missing details <br> with higher probability for training records <br> than unseen records |tbc|


## Mitigations
In the below *raw data* refers to data as provided by the TRE, *training data* refers to data in the format as presented to the model (and hence vulnerable to attack). 

| Name | What does <br>mitigation reduce? | Description |uri | organisationalMeasures |Constraints/Notes |
| ---- |---|---|---|---|---|
|Model access controls | Likelihood <br> $\text{impact}^\dagger$ |Model is deployed to controlled server <br> with access limited to 'whitelisted' addresses <br> and 'query throttling' |tbc| (1) whitelist,<br> (2) throttling rates | 
| DP preprocessing |Impact| Raw data is transformed using a **non-invertible** DP algorithms <br> prior to presentation to model <br> so that data extracted from models cannot be `reverse engineered' | tbc| (1) $\epsilon$<br>(2) $\delta$ (for some implementations) |  privacy guarantees are at *record-level* <br> so data cannot contain multiple records from same person<br><br> Setting of $\epsilon$ is non-trivial$\dagger \dagger$|
| $K$-anonymity |Impact| Preprocessing  introduces a  mapping from 'raw' data to training data <br> which is at least $K \rightarrow 1$ | tbc | $K$ |privacy guarantees are at *record-level* <br> so data cannot contain multiple records from same person|
| PII removal | Impact| Preprocessing removes all direct **or indirect** personal identifiers.<br> Thus even though unique training records could be extracted <br> they could not reasonably be traced to individuals | - | -|Requires agreement with TRE on a dataset-specific basis|
|Differentially Private training | Likelihood| Use of a DP optimiser during model training | tbc |  (1) $\epsilon$<br>(2) $\delta$ (for some implementations) |privacy guarantees are at *record-level* <br> so data cannot contain multiple records from same person <br><br> Setting of $\epsilon$ is non-trivial $\dagger \dagger$|
|Failure of simulated inference attacks|Likelihood| Suite of simulated attacks not statistically better than random guessing <br> **and** same records are not consistently identified by most attacks|tbc|(1) selection of attacks,<br>(2) selection of metrics <br>(e.g. AUC, TPR@lowFPR),<br>(3) $\alpha$- probability threshold <br> for rejecting "null hypothesis" <br> that attack is doing no better than random guessing<br>(4)computational budget <br> (decides what attacks can be run) |'strength' of attacks depends on compute <br><br> typical relies on estimating upper bound of vulnerability<br> by exploiting 'worst-case' scenario of maximum knowledge of datasets|
|Discrebitability of MIAs| Impact|Domingo Ferrer et al (ESORICS) identify a mitigation based only on the nature of the training data<br> regardless of attack.<br>  In some cases, even if it can be inferred that a record must have been present in the training set,<br> that cannot be mapped back to a unique individual.| tbc |(1) $n$: minimum 'arity' of mapping of values to quasi-identifers <br> (2) Organisational attitude?<br>(applies equally to other SDC risks) |data is tabular **and** <br>data is a sample of records available **and** <br>the mapping from 'confidential attribute values' to 'quasi identifer values' <br>is $n:1$ with $n >1$ for all quasi-identifer values| 

Notes:
- $\dagger$ : Arguments about impact for Model Access Controls are that potential release would only be to selected audience, e.g. clinicians within trust boundary. However that could still violate organisational data protection policies 
- $\dagger \dagger$ Privacy guarantee ($\epsilon$) depends on combination of (i) adding noise to groups (batches) of records and (ii) 'clipping' the influence of individual records. Improper settings can satisfy $\epsilon$ values but still leave models/embeddings vulnerable to attack.


## Checks
| Name | What risk(s) does the check measure | Description |uri | organisationalMeasures |Constraints/Notes |
| ---- |---|---|---|---|---|
|Equivalence classes <br> in model outputs<br> for training data | Small group reporting | tbc| - |tolerance to two outputs to be considered identical |can be done explicitly for tree-based models <br> more generally by grouping output probabilities |
|Presence zeros in predicted probabilities| Class Disclosure | A classifier predicting zero probability <br> for a label for group of records <br>  |tbc|min. tolerance for prob(label)<br> to be considerdd non-zero|This is equivalent to a table where one or more cells are empty|
|DoF count|Low Residual Degrees of Freedom| Counts and compares model parameter count <br> vs. training set size.|tbc| minimum residual  DoF | Same as  applied to 'traditional' regression models|
|Generalisation Gap|Membership Inference|Difference between losses on training and test sets.<br> One measure of **over-fitting**.<br>| tbc |max threshold |Typically a **sufficient but not necessary** condition for MIA vulnerability.<br>i.e  reason to reject but not to accept an egress request.|
|Direct model code check|Extraction of explicitly stored training data|Directly accessing model attributes in code <br>e.g. `print(my_SVC.support_vectors_)` |tbc |- |-| 
|Simulated Attack Results| Extraction of implicitly stored <br>MIA/AIA | Results from running simulated attacks. |tbc |choice of metrics<br> |tbc|
---

**Open Question: better to frame "failure of simulated attacks" as a check? or a mitigation with an associated check?**






## Relationship matrices

| <br>Statbarn|Risk <br>   | Extraction of<br> explicitly <br>stored training data | Extraction of<br> implicitly <br>stored training data| Small group <br> reporting | Class <br>disclosure | Low Degrees<br> of Freedom| Membership<br> Inference | Attribute <br> Inference |  
|---          |---         |---                                                    |---                                                   |---                         |---                   |---                        |---                       |---                       |
|Instance-based|           |Y                                                      | -                                                    |Y                           |Y                     |-                          |Y                         |?                         |
|Sequence-based|           |-|Y|?|Y|Y|Y|?|
|Classifiers   |           |-|-|Y|Y|Y|Y|Y|
|Regressors    |           |-|-|?|?|Y|Y|Y|
|Semi-structured<br>outputs||-|-|?|?|Y|?|?|



| <br>Mitigation|Risk <br> Nature | Extraction of<br> explicitly <br>stored training data | Extraction of<br> implicitly <br>stored training data| Small group <br> reporting | Class <br>disclosure | Low Degrees<br> of Freedom| Membership<br> Inference | Attribute <br> Inference |  
|---            |---              |---                                                    |---                                                   |---                         |---                   |---                        |---                       |---                       |
|Model Access Control | Likelihood **and** <br>impact $\dagger$| Y                        |Y                                                     | N                          | N                    | N                         | Y                        | Y                        |
| DP preprocessing.   |Impact.    | Y                                                     | Y                                                    | Y (in theory)              | disputed             | N                         | Y                        |Y                         |
| $K$-anonymity       |Impact.    | Y                                                     | Y                                                    | Y                          | N                    | N                         | Y                        | Y                        |
| PII removal         |Impact.    | Y                                                     | Y                                                    | N                          | N                    | N                         | Y                        |Y                         |
|DP training          |Likelihood | -                                                     | Y                                                    | Y (in theory?)             | disputed             | N                         | Y                        | Y                        |
|Failure of simulated attacks|Likelihood| -                                               |  $Y ^{\dagger\dagger\dagger}$                        | -                          | -                    | -                         | $Y^{\dagger\dagger}$     | $Y^{\dagger\dagger}$     |
|Discreditability.    | Impact    | -                                                     | -                                                    | -                          | -                    | -                         | Y                        | Y                      |



Notes: 
- $\dagger$ : Arguments about impact for Model Access Controls are that potential release would only be to selected audience, e.g. clinicians within trust boundary. However that could still violate organisational data protection policies 
- $\dagger \dagger$ : Membership and Attribute Inference Attacks for 'standard ML' (label prediction for each record is independant of other predictions) are reasonably well understood. The extension to sequence-based models e.g.  Generative AI is a highly active research topic with no current consensus on best practice.
- $\dagger\dagger\dagger$ : the measurement of, and defences against the risk that some prompts may induce Generative AI models to 'regurgitate' training data is a highly active research topic with no current consensus on best practice.

---


# Draft Knowledge Graph for instance-based models
```mermaid 
---
config:
  theme: 'base'
  themeVariables:
    fontSize: 28px
---



block

classDef mitigationStyle fill:lightgreen ;
classDef invisible fill:transparent,stroke:transparent; 
classDef checkStyle fill:orange ;
classDef riskStyle fill:lightpink ;
classDef catStyle fill:lightblue
columns 5
CatA[" Instance-based"] 
class CatA catStyle

space
block:risksblock:3
  columns 1
  block:explicit
    columns 3
    Extraction_Explicit["Extraction of explicitly stored data"] space:2
    space:3
    space:2 DirectQuery["Direct Model Code Check"]  
    space:3
    mitigations["hasMitigations"] space:2
    space:2 MAC["Model Access Controls"]
    space:2 DPE["Differentially Private Preprocessing"]
    space:2 KANON["K-Anonymous preprocessing"]
    space:2 PIIRemoval["Removal of PII"]
    space:2 DPTraining["Use of DP training"]
    space:2 AttackFailure["Simulated Attacks Fail"]
    space:2 Discredit["Attacks can be discredited"]

    Extraction_Explicit -- "hasCheck" --> DirectQuery
    Extraction_Explicit --- mitigations
    mitigations --> MAC
    mitigations --> DPE 
    mitigations --> KANON 
    mitigations --> PIIRemoval 
    class MAC,DPE,KANON,PIIRemoval mitigationStyle
    class checks,mitigations invisible
    class DirectQuery checkStyle
  end
 
  block:smallgroups
    columns 3
    SmallGroups["Reporting on small groups"] space:2 
    space:3
    space:2 EquivClass1[" Equivalence Classes for training records"]  
    space:3
    mitigations1["hasMitigations"] space:2
    space:2 MAC1["Model Access Controls"]
    space:2 DPE1["Differentially Private Preprocessing"]
    space:2 KANON1["K-Anonymous preprocessing"]
    space:2 PIIRemoval1["Removal of PII"]
    space:2 DPTraining1["Use of DP training"]
    space:2 AttackFailure1["Simulated Attacks Fail"]
    space:2 Discredit1["Attacks can be discredited"]


    SmallGroups -- "hasCheck" --> EquivClass1
    SmallGroups --- mitigations1
    %%mitigations1 --> MAC1
    mitigations1 --> DPE1 
    mitigations1 --> KANON1 
    mitigations1 -- "dataset-specific"--> PIIRemoval1
    class DPE1,KANON1,PIIRemoval1 mitigationStyle
    class checks1,mitigations1 invisible
    class EquivClass1 checkStyle
  end
 
  block:classdisclosure
    columns 3
    ClassDisclosure["Class Disclosure"] space:2
    space:3
    space:2 EquivClass2[" Presence of zeros in prediction probabilities"]  
    space:3
    mitigations2["hasMitigations"] space:2
    space:2 MAC2["Model Access Controls"]
    space:2 DPE2["Differentially Private Preprocessing"]
    space:2 KANON2["K-Anonymous preprocessing"]
    space:2 PIIRemoval2["Removal of PII"]
    space:2 DPTraining2["Use of DP training"]
    space:2 AttackFailure2["Simulated Attacks Fail"]
    space:2 Discredit2["Attacks can be discredited"]


    ClassDisclosure -- "hasCheck" --> EquivClass2
    ClassDisclosure --- mitigations2
    %%mitigations2 --> MAC2
    mitigations2 -- "Debatable" --> DPE2 
    %%mitigations2 --> KANON2 
    %%mitigations2 --> PIIRemoval2
    class DPE2 mitigationStyle
    class checks2,mitigations2 invisible
    class EquivClass2 checkStyle
  end
 
  block:mia
    columns 3
    MembershipInference["Vulnerability to Membership Inference"] space:2
    space:3
    space:2 SimAttacks3[" Results from simulated attacks"]  
    space:3
    mitigations3["hasMitigations"] space:2
    space:2 MAC3["Model Access Controls"]
    space:2 DPE3["Differentially Private Preprocessing"]
    space:2 KANON3["K-Anonymous preprocessing"]
    space:2 PIIRemoval3["Removal of PII"]
    space:2 DPTraining3["Use of DP training"]
    space:2 AttackFailure3["Simulated Attacks Fail"]
    space:2 Discredit3["Attacks can be discredited"]


    MembershipInference -- "hasCheck" --> SimAttacks3
    MembershipInference --- mitigations3
    mitigations3 --> MAC3
    mitigations3 --> DPE3 
    mitigations3 --> KANON3 
    mitigations3 --> PIIRemoval3
    mitigations3 --> DPTraining3
    mitigations3 --> AttackFailure3
    mitigations3 --> Discredit3
    class MAC3,DPE3,KANON3,PIIRemoval3,DPTraining3,AttackFailure3,Discredit3 mitigationStyle
    class checks3,mitigations3 invisible
    class SimAttacks3 checkStyle
  end
 
  block:aia
    columns 3
     AttributeInference["Vulnerability to Attribute Inference"] space:2
       space:3
    space:2 SimAttacks4[" Results from simulated attacks"]  
    space:3
    mitigations4["hasMitigations"] space:2
    space:2 MAC4["Model Access Controls"]
    space:2 DPE4["Differentially Private Preprocessing"]
    space:2 KANON4["K-Anonymous preprocessing"]
    space:2 PIIRemoval4["Removal of PII"]
    space:2 DPTraining4["Use of DP training"]
    space:2 AttackFailure4["Simulated Attacks Fail"]
    space:2 Discredit4["Attacks can be discredited"]


    AttributeInference -- "hasCheck" --> SimAttacks4
    AttributeInference --- mitigations4
    mitigations4 --> MAC4
    mitigations4 --> DPE4 
    mitigations4 --> KANON4 
    mitigations4 --> PIIRemoval4
    mitigations4 --> DPTraining4
    mitigations4 --> AttackFailure4
    mitigations4 --> Discredit4
    class MAC4,DPE4,KANON4,PIIRemoval4,DPTraining4,AttackFailure4,Discredit4 mitigationStyle
    class checks4,mitigations4 invisible
    class SimAttacks4 checkStyle

  end
end
CatA -- "hasRisk" --> Extraction_Explicit
CatA -- "hasRisk" --> SmallGroups
CatA -- "hasRisk" --> ClassDisclosure
CatA -- "hasRisk" --> MembershipInference
CatA -- "hasRisk" --> AttributeInference

%%class explicit,smallgroups,classdisclosure,mia,aia riskStyle
class Extraction_Explicit,SmallGroups,FullySpecifiedModel,ClassDisclosure,MembershipInference,AttributeInference riskStyle
```




---
 # Draft Knowledge Graph  for sequence-based models
```mermaid 
---
config:
  theme: 'base'
  themeVariables:
    fontSize: 28px
---



block

classDef mitigationStyle fill:lightgreen ;
classDef invisible fill:transparent,stroke:transparent; 
classDef checkStyle fill:orange ;
classDef riskStyle fill:lightpink ;
classDef catStyle fill:lightblue
columns 5
CatB[" Sequence-based"] 
class CatB catStyle

space
block:risksblock:3
  columns 1
  block:implicit
    columns 3
    Extraction_Implicit["Extraction of implicitly stored data"] space:2
    space:3
    space:2 AttackResults["No current agreed best  practice for checking"]  
    space:3
    mitigations["hasMitigations"] space:2
    space:2 MAC["Model Access Controls"]
    space:2 DPE["Differentially Private Preprocessing"]
    space:2 KANON["K-Anonymous preprocessing"]
    space:2 PIIRemoval["Removal of PII"]
    space:2 DPTraining["Use of DP training"]
    space:2 AttackFailure["Simulated Attacks Fail"]
    space:2 Discredit["Attacks can be discredited"]

    Extraction_Implicit -- "hasCheck" --> AttackResults
    Extraction_Implicit --- mitigations
    mitigations --> MAC
    mitigations --> DPE 
    mitigations --> KANON 
    mitigations --> PIIRemoval 
    class MAC,DPE,KANON,PIIRemoval,DPTraining mitigationStyle
    class checks,mitigations invisible
    class AttackResults checkStyle
  end
 
  block:smallgroups
    columns 3
    SmallGroups["Reporting on small groups"] space:2 
    space:3
    space:2 EquivClass1[" Equivalence Classes for training records"]  
    space:3
    mitigations1["hasMitigations"] space:2
    space:2 MAC1["Model Access Controls"]
    space:2 DPE1["Differentially Private Preprocessing"]
    space:2 KANON1["K-Anonymous preprocessing"]
    space:2 PIIRemoval1["Removal of PII"]
    space:2 DPTraining1["Use of DP training"]
    space:2 AttackFailure1["Simulated Attacks Fail"]
    space:2 Discredit1["Attacks can be discredited"]


    SmallGroups -- "hasCheck" --> EquivClass1
    SmallGroups --- mitigations1
    %%mitigations1 --> MAC1
    mitigations1 --> DPE1 
    mitigations1 --> KANON1 
    mitigations1 -- "dataset-specific"--> PIIRemoval1
    class DPE1,KANON1,PIIRemoval1 mitigationStyle
    class checks1,mitigations1 invisible
    class EquivClass1 checkStyle
  end

 block:lowdof
    columns 3
    LowDegreesOfFreedom["Low Degrees of Freedom "] space:2
    space:3
    space:2 DofCount[" Degrees of freedom count"]  
    space:3
    mitigations1A["hasMitigations"] space:2
    space:2 MAC1A["Model Access Controls"]
    space:2 DPE1A["Differentially Private Preprocessing"]
    space:2 KANON1A["K-Anonymous preprocessing"]
    space:2 PIIRemoval1A["Removal of PII"]
    space:2 DPTraining1A["Use of DP training"]
    space:2 AttackFailure1A["Simulated Attacks Fail"]
    space:2 Discredit1A["Attacks can be discredited"]


    LowDegreesOfFreedom -- "hasCheck" --> DofCount
    LowDegreesOfFreedom --- mitigations1A
    %%mitigations2 --> MAC1A
    %%mitigations2 -- "Debatable" --> DPE2 
    %%mitigations2 --> KANON2 
    %%mitigations2 --> PIIRemoval2
    %%class DPE2 mitigationStyle
    class mitigations1A invisible
    class DofCount checkStyle
  end

  block:classdisclosure
    columns 3
    ClassDisclosure["Class Disclosure"] space:2
    space:3
    space:2 EquivClass2[" Presence of zeros in prediction probabilities"]  
    space:3
    mitigations2["hasMitigations"] space:2
    space:2 MAC2["Model Access Controls"]
    space:2 DPE2["Differentially Private Preprocessing"]
    space:2 KANON2["K-Anonymous preprocessing"]
    space:2 PIIRemoval2["Removal of PII"]
    space:2 DPTraining2["Use of DP training"]
    space:2 AttackFailure2["Simulated Attacks Fail"]
    space:2 Discredit2["Attacks can be discredited"]


    ClassDisclosure -- "hasCheck" --> EquivClass2
    ClassDisclosure --- mitigations2
    %%mitigations2 --> MAC2
    mitigations2 -- "Debatable" --> DPE2 
    %%mitigations2 --> KANON2 
    %%mitigations2 --> PIIRemoval2
    class DPE2 mitigationStyle
    class checks2,mitigations2 invisible
    class EquivClass2 checkStyle
  end
 
  block:mia
    columns 3
    MembershipInference["Vulnerability to Membership Inference"] space:2
    space:3
    space:2 SimAttacks3[" Results from simulated attacks"]  
    space:3
    mitigations3["hasMitigations"] space:2
    space:2 MAC3["Model Access Controls"]
    space:2 DPE3["Differentially Private Preprocessing"]
    space:2 KANON3["K-Anonymous preprocessing"]
    space:2 PIIRemoval3["Removal of PII"]
    space:2 DPTraining3["Use of DP training"]
    space:2 AttackFailure3["Simulated Attacks Fail"]
    space:2 Discredit3["Attacks can be discredited"]


    MembershipInference -- "hasCheck" --> SimAttacks3
    MembershipInference --- mitigations3
    mitigations3 --> MAC3
    mitigations3 --> DPE3 
    mitigations3 --> KANON3 
    mitigations3 --> PIIRemoval3
    mitigations3 --> DPTraining3
    mitigations3 --> AttackFailure3
    mitigations3 --> Discredit3
    class MAC3,DPE3,KANON3,PIIRemoval3,DPTraining3,AttackFailure3,Discredit3 mitigationStyle
    class checks3,mitigations3 invisible
    class SimAttacks3 checkStyle
  end
 
  block:aia
    columns 3
     AttributeInference["Vulnerability to Attribute Inference"] space:2
       space:3
    space:2 SimAttacks4[" Results from simulated attacks"]  
    space:3
    mitigations4["hasMitigations"] space:2
    space:2 MAC4["Model Access Controls"]
    space:2 DPE4["Differentially Private Preprocessing"]
    space:2 KANON4["K-Anonymous preprocessing"]
    space:2 PIIRemoval4["Removal of PII"]
    space:2 DPTraining4["Use of DP training"]
    space:2 AttackFailure4["Simulated Attacks Fail"]
    space:2 Discredit4["Attacks can be discredited"]


    AttributeInference -- "hasCheck" --> SimAttacks4
    AttributeInference --- mitigations4
    mitigations4 --> MAC4
    mitigations4 --> DPE4 
    mitigations4 --> KANON4 
    mitigations4 --> PIIRemoval4
    mitigations4 --> DPTraining4
    mitigations4 --> AttackFailure4
    mitigations4 --> Discredit4
    class MAC4,DPE4,KANON4,PIIRemoval4,DPTraining4,AttackFailure4,Discredit4 mitigationStyle
    class checks4,mitigations4 invisible
    class SimAttacks4 checkStyle

  end
end
CatB -- "hasRisk" --> Extraction_Implicit
CatB -- "hasRisk" --> LowDegreesOfFreedom
CatB -- "hasRisk" --> SmallGroups
CatB -- "hasRisk" --> ClassDisclosure
CatB -- "hasRisk" --> MembershipInference
CatB -- "hasRisk" --> AttributeInference

%%class implicit,smallgroups,classdisclosure,mia,aia riskStyle
class Extraction_Implicit,LowDegreesOfFreedom,SmallGroups,FullySpecifiedModel,ClassDisclosure,MembershipInference,AttributeInference riskStyle
```





---

 # Draft Knowledge Graph  for Classifiers/Regressors
```mermaid 
---
config:
  theme: 'base'
  themeVariables:
    fontSize: 28px
---



block

classDef mitigationStyle fill:lightgreen ;
classDef invisible fill:transparent,stroke:transparent; 
classDef checkStyle fill:orange ;
classDef riskStyle fill:lightpink ;
classDef catStyle fill:lightblue
columns 5
CatC[" Classificatione/Regression Models <br> Excluding fine-tuned foundation models"] 
class CatC catStyle

space
block:risksblock:3
  columns 1
 
  block:smallgroups
    columns 3
    SmallGroups["Reporting on small groups"] space:2 
    space:3
    space:2 EquivClass1[" Equivalence Classes for training records"]  
    space:3
    mitigations1["hasMitigations"] space:2
    space:2 MAC1["Model Access Controls"]
    space:2 DPE1["Differentially Private Preprocessing"]
    space:2 KANON1["K-Anonymous preprocessing"]
    space:2 PIIRemoval1["Removal of PII"]
    space:2 DPTraining1["Use of DP training"]
    space:2 AttackFailure1["Simulated Attacks Fail"]
    space:2 Discredit1["Attacks can be discredited"]


    SmallGroups -- "hasCheck" --> EquivClass1
    SmallGroups --- mitigations1
    %%mitigations1 --> MAC1
    mitigations1 --> DPE1 
    mitigations1 --> KANON1 
    mitigations1 -- "dataset-specific"--> PIIRemoval1
    class DPE1,KANON1,PIIRemoval1 mitigationStyle
    class checks1,mitigations1 invisible
    class EquivClass1 checkStyle
  end

 block:lowdof
    columns 3
    LowDegreesOfFreedom["Low Degrees of Freedom "] space:2
    space:3
    space:2 DofCount[" Degrees of freedom count"]  
    space:3
    mitigations1A["hasMitigations"] space:2
    space:2 MAC1A["Model Access Controls"]
    space:2 DPE1A["Differentially Private Preprocessing"]
    space:2 KANON1A["K-Anonymous preprocessing"]
    space:2 PIIRemoval1A["Removal of PII"]
    space:2 DPTraining1A["Use of DP training"]
    space:2 AttackFailure1A["Simulated Attacks Fail"]
    space:2 Discredit1A["Attacks can be discredited"]


    LowDegreesOfFreedom -- "hasCheck" --> DofCount
    LowDegreesOfFreedom --- mitigations1A
    %%mitigations2 --> MAC1A
    %%mitigations2 -- "Debatable" --> DPE2 
    %%mitigations2 --> KANON2 
    %%mitigations2 --> PIIRemoval2
    %%class DPE2 mitigationStyle
    class mitigations1A invisible
    class DofCount checkStyle
  end

  block:classdisclosure
    columns 3
    ClassDisclosure["Class Disclosure"] space:2
    space:3
    space:2 EquivClass2[" Presence of zeros in prediction probabilities"]  
    space:3
    mitigations2["hasMitigations"] space:2
    space:2 MAC2["Model Access Controls"]
    space:2 DPE2["Differentially Private Preprocessing"]
    space:2 KANON2["K-Anonymous preprocessing"]
    space:2 PIIRemoval2["Removal of PII"]
    space:2 DPTraining2["Use of DP training"]
    space:2 AttackFailure2["Simulated Attacks Fail"]
    space:2 Discredit2["Attacks can be discredited"]


    ClassDisclosure -- "hasCheck" --> EquivClass2
    ClassDisclosure --- mitigations2
    %%mitigations2 --> MAC2
    mitigations2 -- "Debatable" --> DPE2 
    %%mitigations2 --> KANON2 
    %%mitigations2 --> PIIRemoval2
    class DPE2 mitigationStyle
    class checks2,mitigations2 invisible
    class EquivClass2 checkStyle
  end
 
  block:mia
    columns 3
    MembershipInference["Vulnerability to Membership Inference"] space:2
    space:3
    space:2 SimAttacks3[" Results from simulated attacks"]  
    space:3
    mitigations3["hasMitigations"] space:2
    space:2 MAC3["Model Access Controls"]
    space:2 DPE3["Differentially Private Preprocessing"]
    space:2 KANON3["K-Anonymous preprocessing"]
    space:2 PIIRemoval3["Removal of PII"]
    space:2 DPTraining3["Use of DP training"]
    space:2 AttackFailure3["Simulated Attacks Fail"]
    space:2 Discredit3["Attacks can be discredited"]


    MembershipInference -- "hasCheck" --> SimAttacks3
    MembershipInference --- mitigations3
    mitigations3 --> MAC3
    mitigations3 --> DPE3 
    mitigations3 --> KANON3 
    mitigations3 --> PIIRemoval3
    mitigations3 --> DPTraining3
    mitigations3 --> AttackFailure3
    mitigations3 --> Discredit3
    class MAC3,DPE3,KANON3,PIIRemoval3,DPTraining3,AttackFailure3,Discredit3 mitigationStyle
    class checks3,mitigations3 invisible
    class SimAttacks3 checkStyle
  end
 
  block:aia
    columns 3
     AttributeInference["Vulnerability to Attribute Inference"] space:2
       space:3
    space:2 SimAttacks4[" Results from simulated attacks"]  
    space:3
    mitigations4["hasMitigations"] space:2
    space:2 MAC4["Model Access Controls"]
    space:2 DPE4["Differentially Private Preprocessing"]
    space:2 KANON4["K-Anonymous preprocessing"]
    space:2 PIIRemoval4["Removal of PII"]
    space:2 DPTraining4["Use of DP training"]
    space:2 AttackFailure4["Simulated Attacks Fail"]
    space:2 Discredit4["Attacks can be discredited"]


    AttributeInference -- "hasCheck" --> SimAttacks4
    AttributeInference --- mitigations4
    mitigations4 --> MAC4
    mitigations4 --> DPE4 
    mitigations4 --> KANON4 
    mitigations4 --> PIIRemoval4
    mitigations4 --> DPTraining4
    mitigations4 --> AttackFailure4
    mitigations4 --> Discredit4
    class MAC4,DPE4,KANON4,PIIRemoval4,DPTraining4,AttackFailure4,Discredit4 mitigationStyle
    class checks4,mitigations4 invisible
    class SimAttacks4 checkStyle

  end
end

CatC -- "hasRisk" --> SmallGroups
CatC -- "hasRisk" --> ClassDisclosure
CatC -- "hasRisk" --> LowDegreesOfFreedom
CatC -- "hasRisk" --> MembershipInference
CatC -- "hasRisk" --> AttributeInference

%%class implicit,smallgroups,classdisclosure,mia,aia riskStyle
class LowDegreesOfFreedom,SmallGroups,FullySpecifiedModel,ClassDisclosure,MembershipInference,AttributeInference riskStyle
```




 
 ____

