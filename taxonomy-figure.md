# Flowchart for decision making on mitigations and risk for AI models trained in sensitive data
 This is designed as a flowchart but *should* map onto the information needed for extending the statbarn taxomony

```mermaid



flowchart TD

classDef invisible fill:none,stroke:0px,stroke-width:0px,width:0px,height:0px ;
classDef riskStyle fill:lightpink ;
classDef mitigationStyle fill:lightgreen ;
    
classDef constant fill:lightblue,shape:stadium ;
classDef sometext shape:text ;
classDef roundedorange fill:orange,shape:rounded;
classDef roundedred fill:red,shape:rounded;

subgraph MitigationByStandardSDC
   StandardSDC[Application of standard SDC proceedures]
   StandardSDC -->|hasParameter| minthresh["Minimum Group Size"]:::constant
   StandardSDC -->|hasParameter| mindof["Minimum Degrees of Freedom"]:::constant
   StandardSDC -->|hasParameter| nozeros["Zeros indicate class disclosure"]:::constant
   AccComment1@{ shape: text, label: "unlikely to be disclosive
   except small groups in confusion matrix" } --> StandardSDC
end
style MitigationByStandardSDC fill:lightgreen

subgraph MitigationByModelAccessControls
    ModelQueryServer[Deployment on secure server with restricted access ]   
    ModelQueryServer-->|hasParameter| whitelist[list of permitted IPaddresses]:::constant       
    ModelQueryServer-->|hasParameter| throttling[controls on volume of queries]:::constant
    ModelQueryServer-->|mitigatesRisks| mitigatesMAC[Extraction
        Attribute Inference 
        Membership Inference ]
   style mitigatesMAC fill:pink
end
style MitigationByModelAccessControls fill:lightgreen

subgraph MitigationByKAnonymity
    KAnon[preprocessing renders data k-anonymous]
    KAnon -->|hasParameter| K[K]:::constant
    KAnon-->|mitigatesRisks| mitigatesK[Extraction
        Attribute Inference 
        Membership Inference ]
    style mitigatesK fill:pink

    Kcomment@{ shape: text, label: "equivalent to MinThreshold for cell counts." } -.- K
end
style MitigationByKAnonymity fill:lightgreen

subgraph MitigationByPIIRemoval
    Pseudonymised[All PII removed from training data]
    Pseudonymised -->|hasProcess|datasetspecific[data set specific]
    Pseudonymised-->|mitigatesRisks| mitigatesPII[Extraction
        Attribute Inference 
        Membership Inference ]
    style mitigatesPII fill:pink
end
style MitigationByPIIRemoval fill:lightgreen


subgraph MitigationByDPEmbedding
    DPE["`Data is transformed **prior to modelling** using Differentially Private method`"]
    DPE -->|hasParameter|epsilonDPE([epsilon:strength of guarantee]):::constant
    DPE-->|mitigatesRisks| mitigatesDPE[Extraction
        Attribute Inference 
        Membership Inference ]
   style mitigatesDPE fill:pink

    epsiloncomment@{ shape: text, label: "Non-trivial to configure protect against class disclosure" } -.-epsilonDPE
end
style MitigationByDPEmbedding fill:lightgreen

subgraph MitigationByDPOptimiser
    DPO[optimiser is Differentially Private]
    DPO -->|hasParameter|epsilonO[epsilon:strength of guarantee]:::constant
    DPO -->|hasRestriction| duplicatesDPO[Invalid if individuals may have duplicate records]
    style duplicatesDPO fill:lightblue
    epsiloncommentO@{ shape: text, label: "Non-trivial to configure to protect against class disclosure" } -.-epsilonO
end
style MitigationByDPOptimiser fill:lightgreen

subgraph MitigationByAttackInvulnerability
    Invulnerability[Success of Attacks not significantly better than random guessing ]   
    Invulnerability-->|hasParameter| alpha[threshold for probability result occurs by chance]:::constant
end
style MitigationByAttackInvulnerability fill:lightgreen



subgraph RiskOfExtraction
  Extraction(Prompts can trigger regurgitation)
  Extraction --> |hasLikelihood| unquantifiable["difficult to reliably quantify"]
  style unquantifiable shape:circle, fill:pink   
  %%Extraction --> |hasMitigation| Pseudonymised
  %%Extraction --> |hasMitigation| ModelQueryServer[Model Access Controls]  
  extractcomment@{ shape: text, label: "Highly active research field, no meaningful consensus on defence" } -.-unquantifiable
end
style RiskOfExtraction fill:lightpink

subgraph RiskofExplicitStorage
    ExplicitlyStoredData(Model explicitly stores data that can be accessed) 
    ExplicitlyStoredData -->|hasLikelihood| certain["100%"]
    style certain shape:circle, fill:#f11 
    %%ExplicitlyStoredData -->|hasMitigation| KAnon
    %%ExplicitlyStoredData -->|hasMitigation| DPE
    %%ExplicitlyStoredData -->|hasMitigation| Pseudonymised
end
style RiskofExplicitStorage fill:lightpink

subgraph RiskofMembershipInference
    MembershipInference(Behaviour of model for a record supports inference it  was part of the training set)
    MembershipInference -->|hasLikelihood| estimated["`**estimated** by attacks`"]
      style estimated shape:circle, fill:yellow   

    %%MembershipInference --> |hasMitigation|KAnon
    %%MembershipInference -->|hasMitigation| Pseudonymised
    MembershipInference ---->|hasMitigation| DPO
    MembershipInference ---->|hasMitigation| Invulnerability
end
style RiskofMembershipInference fill:lightpink

subgraph RiskofAttributeInference
    AttributeInference(Behaviour of model for different completions of a partial record supports inference of missing values)
    AttributeInference -->|hasLikelihood| estimated2["`**estimated** by attacks`"]
      style estimated2 shape:circle, fill:yellow   

    %%AttributeInference --> |hasMitigation|KAnon
    %%AttributeInference -->|hasMitigation| Pseudonymised
    AttributeInference ---->|hasMitigation| DPO
    AttributeInference ---->|hasMitigation| Invulnerability
end
style RiskofAttributeInference fill:lightpink

subgraph RiskOfSmallGroups
  SmallGroups(Model partitions data so that it effectivly reports on small groups of records)
  SmallGroups -->|HasLikelihood| calculatedSG["`**calculated** by attack`"]
end
style RiskOfSmallGroups fill:lightpink

subgraph RiskOfClassDisclosure
  ClassDisclosure(Model reports some values do not occur - classification -- or upper/lower bounded or some small groups of records --regression f)
  ClassDisclosure -->|HasLikelihood| calculatedCD["`**calculated** by attack`"]
end
style RiskOfClassDisclosure fill:lightpink


subgraph RiskOfFullySpecifiedModel
  FullySpecified[Low Residual Degrees of Freedom means model is effectively a lookup table]
    FullySpecified -->|HasLikelihood| calculatedFS["`**Calculated** by attack`"]
          style calculatedFS shape:circle, fill:lightgreen   
    FScomment@{ shape: text, label: "Standard SDC process to measure DoF" } -.- calculatedFS
end
style RiskOfFullySpecifiedModel fill:lightpink

%% Here's the main block
A{Type of egress?} 
A -->|trained model| Destination{Destination of egress}

A -->|performance metrics| StandardSDC

%% Destination of egress
Destination -->|Servers with access controls| ModelQueryServer
Destination ---->|Outside World| ImpactBasedMitigation{Is the mitigation based on the impact of model leakage}
ImpactBasedMitigation -->|Yes| NoDups{Only one record per individual}
NoDups --> Yes:::invisible
Yes -->|hasSourceOfEvidence| MitigationByDPEmbedding
Yes -->|hasSourceOfEvidence| MitigationByKAnonymity
Yes -->|hasSourceOfEvidence| MitigationByPIIRemoval
NoDups --> No:::invisible
No -->|hasSourceOfEvidence| MitigationByPIIRemoval

%%personal data going to the real world
ImpactBasedMitigation ---->|No| Type{Basis of prediction}
%% instance based - no
Type -->|Comparison to stored data| CatA([CategoryA :Instance-based
                              e.g. Support Vector Machines,
                              k-Nearest Neighbours]):::roundedred

CatA -->|hasRisk| ExplicitlyStoredData
certain -->Refuse
style Refuse shape:stadium, fill:red

%% Gen AI - No
Type -->|SequenceBased| CatB([Category B
                              e.g. Generative AI]):::roundedred ---|hasRisk| Extraction 
unquantifiable-->Refuse


%% Cat C Maybe 
Type ------> |Independent per record| CatC([Category C
                              Independent
                              predictions for each input]):::roundedorange
CatC ----->LabelType{Type of prediction}
LabelType -->|semi-structured|Segmentation([C3: Segmentation, Regions of Interest]):::roundedorange
Segmentation -->|hasRisk| MembershipInference

LabelType -->|unordered|Classification([C2: Classification Models]):::roundedorange

LabelType -->|ordered|Regression([C1: Regression Models]):::roundedorange


Classification -->Common[Supervised Learner]:::roundedorange
Regression -->|Nature of Variables|SpecialReg{All independent variables continuous, cannot build piecewise models}
SpecialReg  -->|No| Common
SpecialReg -->|Yes| FullySpecified

Common --> |hasRisk| MembershipInference
Common -->|hasRisk| AttributeInference
Common -->|hasRisk| SmallGroups
Common -->|hasRisk| ClassDisclosure
Common -->|hasRisk| FullySpecified

```




## Takeaway points:
A whole lot of links between risks and mitigation are missed (see pink boxes within some green boxes)
model going to ControlledAcessServer -> fine
Safety case is that preprocessing transforms records so they would be ok to release ==> set if possible evidence that could be collected
Otherwise:

- instance based models (CatA) ==>Likelihood of attack via explicitly stored data is 100%  => NO
- Generative AI (CatB) ==> Risk of 'Extraction' (Regurgitation) ==>no consensus on reliable measure or defence ==> NO
- independent predictions for each record (catC) :  one edge case for simplistic regression otherweise range of risks and defences

### This does not distinguish between record level and identity level risk (e.g. person has multiple records for different hospital visits )
