# Counter-fitting Word Vectors to Linguistic Constraints
Nikola Mrkšić (nm480@cam.ac.uk)

This repository contains the code and data for the method presented in [Counter-fitting Word Vectors to Linguistic Constraints](https://arxiv.org/pdf/1603.00892.pdf). The word vectors which achieve the (present) state of the art (0.74) on the SimLex-999 dataset are included in this repository.


###Configuring the Tool

The counter-fitting tool reads all the experiment config parameters from the ```experiment_parameters.cfg``` file in the root directory. An alternative config file can be provided as the first (and only) argument to ```counterfitting.py```. 

The config file specifies:
* the location of the initial word vectors ```[default: word_vectors/glove.txt]```
* the vocabulary to be used ```[default: lingustic_constraints/vocabulary.txt]``` 
* the sets of linguistic constraints to be injected into the vector space. The ```linguistic_constraints``` directory contains the synonymy (PPDB 2.0) and antonymy (WordNet and PPDB 2.0) constraints used in our experiments. 
* optionally, one can also specify the location of a dialogue domain ontology (in the DSTC format). This ontology will be used to infer additional antonymy constraints between slot values. The ```linguistic_constraints``` directory contains the two dialogue ontologies (DSTC2, DSTC3) used in our experiments. 

The config file also specifies the six hyperparameters of the counter-fitting procedure (set to their default values in ```experiment_parameters.cfg```). 

The results directory also contains the SimLex-999 dataset (Hill et al., 2014), required to perform the evaluation. 


###Running Experiments

```python counterfitting.py experiment_parameters.cfg```

Running the experiment loads the word vectors specified in the config file and counter-fits them to the provided linguistic constraints. The procedure prints the updated word vectors to the results directory as ```counter_fitted_vectors.txt``` (one word vector per line). The produced ranking and the gold standard ranking for the SimLex-999 pairs are also printed to the results directory. 

The word_vectors directory contains the (zipped) GloVe and Paragram-300-SL999 vectors constrained to our vocabulary (these need to be unzipped before the experiments are run). The high-scoring vectors for SimLex-999 can also be found in ```word_vectors/counter-fitted-vectors.txt.zip``` (or reproduced by applying counter-fitting to Paragram vectors).


###References

The counter-fitting paper:
```
@InProceedings{mrksic:2016:naacl,
  author    = {Nikola Mrk\v{s}i\'c and Diarmuid {\'O S\'eaghdha} and Blaise Thomson and Milica Ga\v{s}i\'c 
  			   and Lina Rojas-Barahona and Pei-Hao Su and David Vandyke and Tsung-Hsien Wen and Steve Young},
  title     = {Counter-fitting Word Vectors to Linguistic Constraints},
  booktitle = {Proceedings of HLT-NAACL},
  year      = {2016},
}
```

If you are using PPDB 2.0 (Pavlick et al., 2015) or WordNet (Miller, 1995) constraints, please cite these papers. If you are using the provided pre-trained vectors, please cite (Pennington et al., 2014) for GloVe vectors and (Wieting et al., 2015) for Paragram-SL-999 vectors. 
