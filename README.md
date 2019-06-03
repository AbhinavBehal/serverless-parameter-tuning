# SOFTENG 751 - Parallel Machine Learning (Group 5)

## Description

This project attempts to assess the performance of several hyperparameter tuning algorithms:
- Grid Search
- Random Search
- Successive Halving Algorithm (SHA)
- Asynchronous Successive Halving Algorithm (ASHA)

Grid and random search have been implemented using the [scikit-learn](https://scikit-learn.org/stable/) machine learning 
package. While SHA and ASHA have been implemented from scratch, using [this](https://arxiv.org/abs/1810.05934) research paper.
All implementations run in parallel by default, with grid search, random search and SHA using process-based parallelism
and ASHA using asynchronous function calls to an AWS lambda.

The dataset used can be found [here](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package).

Results obtained from the scripts in the `benchmarking/` folder can be found in `results.xlsx`. 
This spreadsheet compares the performance of all four tuning algorithms, and also contains 
data specific to ASHA and the effects of its various input parameters.

## Setup
- Install Python 3
- Clone this repository
- Run `pip install -r requirements.txt`
- Note
    - If you want to run the ASHA algorithm, you will need to modify `tuning/asha.py` to access
    your own AWS Lambda. The code for this lambda function can be found in
    `lambda/run_xgboost.py`. You will also need to place your AWS credentials in `~/.aws/credentials`,
    and set a default region in `~/.aws/config`.
    - Additionally, you will need to include a copy of your training data in the `lambda/` folder
    when uploading to AWS.
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
