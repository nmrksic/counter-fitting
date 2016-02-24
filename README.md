# Counter-fitting Word Vectors to Linguistic Constraints
Nikola Mrkšić, nm480@cam.ac.uk

This repository contains the code and input data for the counter-fitting algorithm presented in (Mrkšić, Ò Séaghdha et al., 2016). The produced word vectors which achieve the present state of the art (0.74) on the SimLex-999 dataset are also included in this repository. 


###Configuring the Tool

The counter-fitting tool reads all the experiment config parameters from the experiment_parameters.cfg file in the root directory. An alternative config file can be provided as the first (and only) argument to counterfitting.py 

The config file specifies the locations of the starting word vectors, the vocabulary to be used and the sets of linguistic constraints to be injected into the vector space. Optionally, the user can also specify the location of a dialogue domain ontology in the DSTC format, which will be used to infer additional antonymy constraints. The config file also specifies the six hyperparameters of the counter-fitting procedure (set to default values in experiment_parameters.cfg). 

The linguistic_constraints directory contains the synonymy and antonymy constraints used in the paper. These are drawn from WordNet and the PPDB 2.0 XXXL packages. The directory also contains the vocabulary used in our experiments and the SimLex-999 dataset, required to perform the evaluation. 


###Running Experiments

```python counterfitting.py experiment_parameters.cfg```

Running the experiment loads the word vectors specified in the config file and counter-fits them to the provided linguistic constraints. If no .cfg file is specified the default one is used. The procedure prints the final vectors to the results directory as a .txt file (one word vector per line). The produced ranking and the gold standard ranking for the SimLex-999 pairs are also printed to the results directory. 

