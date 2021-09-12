# orfHMM
A program to identify the translational regulation elements in mRNA

# Overview
The program contains two modules: `orfHMM_EM.py` and `orfHMM_Viterbi.py`. They have to run in order to achieve the final results. Example codes and outputs are given in the folder `Simulation Results` with Jupyter Notebook.

`orfHMM_EM.py`: Refining both transition probability and emission probability with EM algorithm. There are three transition parameter in twenty-one states model and one transition parameter in ten states case. Number emission probability depends on number of hidden states and updating format.

`orfHMM_Viterbi.py`: Backtrack the hidden states according to the Viterbi algorithm.

# Package Dependency
For `orfHMM_EM.py` and `orfHMM_Viterbi.py`:
* `numpy`
* `scipy`

Addition requirnment for generating results in `Simulation Results`:
* `matplotlib`
* `pandas`
* `seaborn`

# Install
```
git clone https://github.com/shimlab/orfHMM.git
```

# Module
## `orfHMM_EM.py`
By applying ` EM_iter(RNA_data, observed_data, E, trans_init, alpha_init, beta_init, epsilon, max_iter, fixed, stop_codon_list, model1)` from `orfHMM_EM.py`, both transition probability and emission probability will be updated according to log likelihood.
```
Inputs:
    RNA_data: a list of lists. Each inner list represents one RNA sequence (single string/char)
    observed_data: a list of lists. Each inner list represents ribosome counts (scalar)
    E: a list of scalars. Normalization factors (scalar)
    trans_init: dictionary. key: start codon, value: a list of transition parameters (for twenty-one states) or a scalar (for ten states)
    alpha_init: a list of scalars. Initial alpha values for different states (NB parameter)
    beta_init: a list of scalars. Initial beta values for different states (NB parameter)
    epsilon: scalar. Stop iteration if difference between two log likelihood smaller than this
    max_iter: int. Max number of iteration times
    fixed: boolean (True/False). Indicates beta fixed or not (False represents update both alpha and beta, vice versa)
    stop_codon_list: a list of stop codons (string)
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    
Outputs:
    trans: dictionary. Updated transition probability for different start codons
    alpha_list: a list of scalars. Updated alpha values for different states
    beta_list: a list of scalars. Updated beta values for different states
```

## `orfHMM_Viterbi.py`
By applying `viterbi_sequence(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1)` from `orfHMM_Viterbi.py`, hidden states can be revealed. 
```
Inputs:
    RNA_data: a list of lists. Each inner list represents one RNA sequence (single string/char)
    observed_data: a list of lists. Each inner list represents ribosome counts (scalar)
    alpha_list: a list of alpha values (NB parameter)
    beta_list: a list of beta values (NB parameter)
    E: a list of scalars. Normalization factors (scalars)
    trans: a dictionary. key: start codon, value: a list of transition parameters (for twenty-one states) or a scalar (for ten states)
    stop_codon_list: a list of stop codons (string)
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)

Outputs:
    output_list: a list of lists. Each inner list represents the predicted hidden states for one RNA sequence
```


