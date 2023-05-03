# Causal Knowledge Tracing
The code for the EDM 2023 paper - A Conceptual Model for End-to-End Causal Discovery in Knowledge Tracing

## Setups
The required environment is as bellow:  
- Unix-based system 
- Python 3.9+
- PyTorch 1.12.1 
- Numpy 1.23.1
- Wandb 0.13.4
- Pandas 1.5.0
- SciPy 1.9.1
- HuggingFace (Transformers) 4.18.0

## Repository Overview 

```/data/``` - Repository containing the dataset

```/serialized_torch/``` - Intermediate data from the prepocessed dataset

```predict_graph.py``` - Training Causal KT

```construct_solution.py``` - Creating adjacency matrix for submission

```/submissions/final``` - Contains the model and the adjacency matrix for submission on the private leaderboard. 

## Running the Causal GRU Model
First install all dependecies of the projects below is the pip methodology. (Similiar approaches exist for conda or other package management systems).

```
pip3 install -r requirements.txt
```


Here is an example how to train the Causal GRU Model with a `learning_rate` of 0.01.
```
python3 predict_graph.py -L 1e-3
```
For a list of all possible hyperparameters see:
```
python3 predict_graph.py -h
```
The output of this is a model `.pt` file which contains a learned P and L matrix. By default, this is saved into the `saved_models` directory. 

To construct the construct ordering adjacency matrix from this model we use `construct_solution.py`. Here is an example how to create a submission file. You do not need to include .pt.
```
python3 construct_solution.py -f <Your Model File>
```

The output of this script is a zip file containing the `.npy` casual order adjency matrix.

Contact: ml4ed @ UMass Amherst
- Nischal Ashok Kumar (nashokkumar@umass.edu)
- Wanyong Feng (wanyongfeng@umass.edu)
- Jaewook Lee (jaewooklee@umass.edu)
- Hunter McNichols (wmcnichols@umass.edu)
- Aritra Ghosh (arighosh@cs.umass.edu)
