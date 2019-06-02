# 751 Project - Parallel Machine Learning

## Setup
- Install Python 3
- Clone this repository
- Run `pip install -r requirements.txt`
- Note
    - If you want to run the ASHA algorithm, you will need to modify `tuning/asha.py` to access
    your own AWS Lambda. The code for this lambda function can be found in
    `lambda/run_xgboost.py`. You will also need to place your AWS credentials in `~/.aws/credentials`,
    and set a default region in `~/.aws/config`.
    - Refer to [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html)
    for more details about the credential and configuration files.


## Usage
- `python main.py -a [algorithm] -p [parameters]`


## Algorithms and Parameters

### Grid Search
- `-a grid`
#### Parameters
```
{
    "max_samples": max. number of samples in each hyperparameter list (increases total combinations),
    "cv": number of folds for cross-validation
}
```

#### Example
`python main.py -a grid -p '{"max_samples": 4, "cv": 3}'`

---

### Random Search
- `-a random`
#### Parameters
```
{
    "n_iter": number of random configurations (iterations) to evaluate,
    "cv": number of folds for cross-validation
}
```
#### Example
`python main.py -a random -p '{"n_iter": 10, "cv": 3}'`

---

### Successive Halving Algorithm (SHA)
- `-a sha`
#### Parameters
```
{
    "n_configs": number of configurations to evaluate,
    "min_r": minimum resources (boosting rounds) given to each configuration,
    "max_r": maximum resources (boosting rounds) given to each configuration,
    "reduction_factor": amount of configurations to be dropped per iteration (2 = reduce by half)
    "cv": number of folds for cross-validation
}
```
#### Example
`python main.py -a sha -p '{"n_configs": 32, "min_r": 1, "max_r": 32, "reduction_factor": 2, "cv": 3}'`

---

### Asynchronous Successive Halving Algorithm (ASHA)
- `-a asha`
#### Parameters
```
{
    "n_workers": number of workers allocated to evaluate the configurations in parallel,
    "min_r": minimum resources (boosting rounds) given to each configuration,
    "max_r": maximum resources (boosting rounds) given to each configuration,
    "reduction_factor": amount of configurations to be dropped per iteration (2 = reduce by half),
    "early_stopping_rounds": number of rounds in which the error must decrease
    "cv": number of folds for cross-validation
}
```
#### Example
`python main.py -a asha -p '{"n_workers": 100, "min_r": 1, "max_r": 64, "reduction_factor": 4, "cv": 3}'`

---

## Note
If running the example commands in windows command line, you will need to escape the double quotes, i.e.

`python main.py -a random -p '{"n_iter": 10, "cv": 3}'`

Becomes

`python main.py -a random -p '{\"n_iter\": 10, \"cv\": 3}'`
