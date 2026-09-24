```mermaid
%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%

flowchart TD

subgraph MitigationByKAnonymity
    KAnon[preprocessing renders data k-anonymous]
    KAnon -->|hasParameter| K([K])
    style K shape:circle, fill:lightblue
    Kcomment@{ shape: text, label: "equivalent to MinThreshold for cell counts" }
    Kcomment -.- K
end
style MitigationByKAnonymity fill:lightgreen

subgraph MitigationByPIIRemoval
    Pseudonymised[All PII removed from training data]
    Pseudonymised -->|hasProcess|datasetspecific[data set specific]
end
style MitigationByPIIRemoval fill:lightgreen


subgraph MitigationByDifferentialPrivacy
    DP[Embedding or optimiser is Differentially Private]
    DP -->|hasParameter|eta[eta:strength of guarantee]
    style eta shape:circle, fill:lightblue
    etacomment@{ shape: text, label: "Non-trivial to configure if duplicates may be present" }
    etacomment -.-eta
end
style MitigationByDifferentialPrivacy fill:lightgreen

subgraph MitigationByAttackInvulnerability
    Invulnerability[Success of Attacks not significantly better than random guessing ]   
    Invulnerability-->|hasParameter| alpha[threshold for probability result occurs by chance]
    style alpha shape:circle, fill:lightblue     
end
style MitigationByAttackInvulnerability fill:lightgreen

subgraph MitigationByModelAccessControls
    ModelQueryServer[Deployment on secure server with restricted access ]   
    ModelQueryServer-->|hasParameter| whitelist[list of permitted IPaddresses]
    style whitelist shape:circle, fill:lightblue     
    ModelQueryServer--> |hasParameter| throttling[controls on volume of queries]
    style throttling shape:circle, fill:lightblue     
end
style MitigationByModelAccessControls fill:lightgreen

subgraph RiskOfExtraction
  Extraction(Prompts can trigger regurgitation)
  Extraction --> |hasLikelihood| unquantifiable["difficult to reliably quantify"]
  style unquantifiable shape:circle, fill:pink   
  Extraction --> |hasMitigation| Pseudonymised
  Extraction --> |hasMitigation| ModelQueryServer[Model Access Controls]  
end
style RiskOfExtraction fill:lightpink

subgraph RiskofExplicitStorage
    ExplicitlyStoredData(Model explicitly stores data that can be accessed) 
    ExplicitlyStoredData -->|hasLikelihood| certain["100%"]
    style certain shape:circle, fill:#f11 
    ExplicitlyStoredData -->|hasMitigation| KAnon
    ExplicitlyStoredData -->|hasMitigation| DP
    ExplicitlyStoredData -->|hasMitigation| Pseudonymised
end
style RiskofExplicitStorage fill:lightpink

subgraph RiskofMembershipInference
    MembershipInference(Behaviour of model for a record supports inference it  was part of the training set)
    MembershipInference -->|hasLikelihood| estimated[estimated by attacks]
      style estimated shape:circle, fill:yellow   

    MembershipInference --> |hasMitigation|KAnon
    MembershipInference -->|hasMitigation| Pseudonymised
    MembershipInference -->|hasMitigation| DP
    MembershipInference -->|hasMitigation| Invulnerability
end
style RiskofMembershipInference fill:lightpink

subgraph RiskofAttributeInference
    AttributeInference(Behaviour of model for different completions of a partial record supports inference of missing values)
    AttributeInference -->|hasLikelihood| estimated2[estimated by attacks]
      style estimated2 shape:circle, fill:yellow   

    AttributeInference --> |hasMitigation|KAnon
    AttributeInference -->|hasMitigation| Pseudonymised
    AttributeInference -->|hasMitigation| DP
    AttributeInference -->|hasMitigation| Invulnerability
end
style RiskofAttributeInference fill:lightpink



%%here's the main block
A{Type of egress?} -->|trained model| Type{Type of Model}
%%
A -->|performance metrics| AccMetrics[standard SDC rules]
style AccMetrics fill:green
AccComment1@{ shape: text, label: "unlikely to be disclosive
except small groups in confusion matrix" }
AccComment1 --> AccMetrics
%%
Type -->|Instance Based| CatA([CategoryA
                              e.g. Support Vector Machines,
                              k-Nearest Neighbours])
style CatA fill:red
CatA --->|hasRisk| ExplicitlyStoredData
%%
Type --> |SequenceBased| CatB([Category B
                              e.g. Generative AI])
style CatB  fill:red
CatB ---- hasRisk ------> Extraction
%%

 
Type --> |Independent| CatC([Category C
                              Independent
                              predictions for each input])
style CatC shape:rounded,fill:orange
CatC -->LabelType{Type of prediction}
LabelType -->|ordered|Regression([C1: Regression Models])
style Regression shape:rounded,fill:orange
LabelType -->|unordered|Classification([C2: Classification Models])
style Classification shape:rounded,fill:orange
LabelType -->|semi-structured|Segmentation([C3: Segmentation, Regions of Interest])
style Segmentation shape:rounded,fill:orange

Regression --> |hasRisk| MembershipInference
Classification --> |hasRisk| MembershipInference
Regression -->|hasRisk| AttributeInference
Classification -->|hasRisk| AttributeInference
