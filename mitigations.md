# Potential Mitigations that might be used as part of a Safety Argument
## Executive Summary
In other  documents we have described:
- differentyep of maxchine learning projects
- the privacy risks that may occur for different types of ML models

### This version:
- 0.1
- Authors Jim Smith, Simon Rogers + ...
- Jan 2026

## Options
- model query server: **reduces likelihood**
- you can demonstrate the training data has been pseudonomised to the extent you would publish it (down a tier in the Turing categorisation): **affects impact**
- risk likelihood measurement (eg running MIA attacks)n- not really a mitigation:
-    - MIAs are no different to random guessing? ON AVERAGE
     - individual records are or are not vulnerable to all MIA sattacks
     - 
- training with differential private optimisers : **reduces likelihood of MIA/being able to make inferenes at individual row level**,
-   for (explain caveats= doesn;t redice likelihood to all risks class disclosure)
- make training data more anonymous **reduces impact**
- better hyper-parameter choice  (e.g. min_samples_leaf in tree based models) : **reduces likelihood**
- preprocessing could be many-to-one or not invertible
-  --- ongoing research question: (how quickly does output change wrt input wrt distance between records in the input space)

All based on risk matrices: if you say you impact is lower then maybe you don't need to run attacks
