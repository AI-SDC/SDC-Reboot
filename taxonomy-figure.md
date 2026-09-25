# Flowchart for decision making on mitigations and risk for AI models trained in sensitive data
 This is designed as a flowchart but *should* map onto the information needed for extending the statbarn taxomony

```mermaid


flowchart LR
subgraph MitigationByModelAccessControls
    ModelQueryServer[Deployment on secure server with restricted access ]   
    ModelQueryServer-->|hasParameter| whitelist[list of permitted IPaddresses]
    style whitelist shape:circle, fill:lightblue     
    ModelQueryServer-->|hasParameter| throttling[controls on volume of queries]
    style throttling shape:circle, fill:lightblue     
    ModelQueryServer-->|mitigatesRisks| mitigatesMAC[Extraction
        Attribute Inference 
        Membership Inference ]
   style mitigatesMAC fill:pink
end
style MitigationByModelAccessControls fill:lightgreen

subgraph MitigationByKAnonymity
    KAnon[preprocessing renders data k-anonymous]
    KAnon -->|hasParameter| K([K])
    style K shape:circle, fill:lightblue
        KAnon-->|mitigatesRisks| mitigatesK[Extraction
        Attribute Inference 
        Membership Inference ]
    style mitigatesK fill:pink

    Kcomment@{ shape: text, label: "equivalent to MinThreshold for cell counts" }
    Kcomment -.- K
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
    DPE -->|hasParameter|eta[eta:strength of guarantee]
    style eta shape:circle, fill:lightblue
    DPE-->|mitigatesRisks| mitigatesDPE[Extraction
        Attribute Inference 
        Membership Inference ]
   style mitigatesDPE fill:pink

    etacomment@{ shape: text, label: "Non-trivial to configure if duplicates may be present" }
    etacomment -.-eta
end
style MitigationByDPEmbedding fill:lightgreen

subgraph MitigationByDPOptimiser
    DPO[optimiser is Differentially Private]
    DPO -->|hasParameter|etaO[eta:strength of guarantee]
    style etaO shape:circle, fill:lightblue
    etacommentO@{ shape: text, label: "Non-trivial to configure if duplicates may be present" }
    etacommentO -.-etaO
end
style MitigationByDPOptimiser fill:lightgreen

subgraph MitigationByAttackInvulnerability
    Invulnerability[Success of Attacks not significantly better than random guessing ]   
    Invulnerability-->|hasParameter| alpha[threshold for probability result occurs by chance]
    style alpha shape:circle, fill:lightblue     
end
style MitigationByAttackInvulnerability fill:lightgreen



subgraph RiskOfExtraction
  Extraction(Prompts can trigger regurgitation)
  Extraction --> |hasLikelihood| unquantifiable["difficult to reliably quantify"]
  style unquantifiable shape:circle, fill:pink   
  %%Extraction --> |hasMitigation| Pseudonymised
  %%Extraction --> |hasMitigation| ModelQueryServer[Model Access Controls]  
  extractcomment@{ shape: text, label: "Highly active research field, no meaningful consensus on defence" }
  extractcomment -.-unquantifiable
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
    MembershipInference -->|hasMitigation| DPO
    MembershipInference -->|hasMitigation| Invulnerability
end
style RiskofMembershipInference fill:lightpink

subgraph RiskofAttributeInference
    AttributeInference(Behaviour of model for different completions of a partial record supports inference of missing values)
    AttributeInference -->|hasLikelihood| estimated2["`**estimated** by attacks`"]
      style estimated2 shape:circle, fill:yellow   

    %%AttributeInference --> |hasMitigation|KAnon
    %%AttributeInference -->|hasMitigation| Pseudonymised
    AttributeInference -->|hasMitigation| DPO
    AttributeInference -->|hasMitigation| Invulnerability
end
style RiskofAttributeInference fill:lightpink

subgraph RiskOfSmallGroups
  SmallGroups(Model partitions data so that it effectivly reports on small groups of records)
  SmallGroups -->|HasLikelihood| calculatedSG["`**calculated** by attack`"]
end
style RiskOfSmallGroups fill:lightpink

subgraph RiskOfClassDisclosure
  ClassDisclosure(Model reports some values do not occur - classification -- or upper/lower bounded  --regression-- for some small groups of records)
  ClassDisclosure -->|HasLikelihood| calculatedCD["`**calculated** by attack`"]
end
style RiskOfClassDisclosure fill:lightpink


subgraph RiskOfFullySpecifiedModel
  FullySpecified[Low Residual Degrees of Freedom means model is effectively a lookup table]
    FullySpecified -->|HasLikelihood| calculatedFS["`**Calculated** by attack`"]
          style calculatedFS shape:circle, fill:lightgreen   
    FScomment@{ shape: text, label: "Standard SDC process to measure DoF" }
    FScomment -.- FullySpecified
end
style RiskOfFullySpecifiedModel fill:lightpink

%%here's the main block
A{Type of egress?} -->|trained model| Destination{Destination of egress}

A -->|performance metrics| AccMetrics[standard SDC rules]
style AccMetrics fill:lightgreen
AccComment1@{ shape: text, label: "unlikely to be disclosive
except small groups in confusion matrix" }
AccComment1 --> AccMetrics

%% Destination of egress
Destination -->|Servers with access controls| ModelQueryServer

Destination -->|Outside World| ImpactBasedMitigation{Is the mitigation based on the impact of model leakage}
ImpactBasedMitigation -->|Yes| Rationale{Documented Evidence}
Rationale -->|ProvidedBy| MitigationByDPEmbedding
Rationale -->|ProvidedBy| MitigationByKAnonymity
Rationale -->|ProvidedBy| MitigationByPIIRemoval

%%personal data going to the real world
ImpactBasedMitigation -->|No| Type{Type of Model}
Type -->|Instance Based| CatA([CategoryA
                              e.g. Support Vector Machines,
                              k-Nearest Neighbours])
%% instance based - no
style CatA fill:red
CatA -->|hasRisk| ExplicitlyStoredData
certain -->Refuse
style Refuse shape:stadium, fill:red

%% Gen AI - No
Type -->|SequenceBased| CatB([Category B
                              e.g. Generative AI]) ---|hasRisk| Extraction 
unquantifiable-->Refuse
style CatB  fill:red


%% CAt C Maybe 
Type --> |Independent| CatC([Category C
                              Independent
                              predictions for each input])
style CatC shape:rounded,fill:orange
CatC -->LabelType{Type of prediction}

LabelType -->|unordered|Classification([C2: Classification Models])
style Classification shape:rounded,fill:orange

LabelType -->|ordered|Regression([C1: Regression Models])
style Regression shape:rounded,fill:orange
LabelType -->|semi-structured|Segmentation([C3: Segmentation, Regions of Interest])
style Segmentation shape:rounded,fill:orange

Classification -->Common[ML vulnerabilities]
Regression -->|Nature of Variables|SpecialReg{All independent variables continuous, cannot build piecewise models}
SpecialReg  -->|No| Common
SpecialReg -->|Yes| FullySpecified

Common --> |hasRisk| MembershipInference
Common -->|hasRisk| AttributeInference
Common -->|hasRisk| SmallGroups
Common -->|hasRisk| ClassDisclosure
Common -->|hasRisk| FullySpecified
