# -*- coding: utf-8 -*-
"""
Created on Sun Feb 6 22:12:29 2021

@author: Sophie Sebille

This file contains ???
"""


## Load libraries
from feature_extractor import fingerprint_features
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from rdkit.Chem import Descriptors


## Functions
def read_split_data(data_path, model_num, train_ratio, validation_ratio, test_ratio):
    """Split original data into train data validation data, and test data.
    :param data_path: str, path to the a CSV data file
    :param model_num: int, moel of interest (1 or 2)
    :param train_ratio: float, proportion of the original data for trainset, must be from 0 to 1
    :param validation_ratio: float, proportion of the original data for validationset, must be from 0 to 1
    :param test_ratio: float, proportion of the original data for testset, must be from 0 to 1
    :return x_train:
    :return y_train:
    :return x_validation:
    :return y_validation:
    :return x_test:
    :return y_test:
    """

    sum_ratios = train_ratio+validation_ratio+test_ratio
    if sum_ratios != 1:
    	return print("Sum of the three ratios should be 1")
    
    # Read data
    dataset = read_csv(data_path)
    array = dataset.values
    Y = dataset['P1'].astype("category")
    if model_num == 1:
    	X = np.asarray(list(map(fingerprint_features, array[:,2])))
    elif model_num == 2:
		#Extract descriptors
    	X = dataset['smiles']
    	X['mol'] = X['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    	X['tpsa'] = X['mol'].apply(lambda x: Descriptors.TPSA(x))
    	X['mol_w'] = X['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    	X['num_valence_electrons'] = X['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    	X['num_heteroatoms'] = X['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    	X = X.drop(columns=['smiles', 'mol'])

    else:
    	return print("A model number must be choosen between 1 or 2")

    # We first split the dataset to create the train subset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - train_ratio, random_state=1)
    
    # We split the test dataset into a final test subset and validation subset
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), random_state = 1, shuffle = False)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def train(x_train, y_train):
    """Train with Grid-search cross validation to find the best hyperparameter
    :param x_train: trainset features
    :param y_train: trainset label
    :return best_estimator: Scikit-learn estimator with the best hyper parameter
    :return best_score: best accuracy score
    :return best_param: dict, best hyper parameter
    """

    rf = GridSearchCV(RandomForestClassifier(), cv = 8, param_grid = {"n_estimators": np.linspace(50, 1000, 50).astype('int')}, scoring = 'neg_mean_absolute_error', n_jobs = -1)
    rf.fit(x_train, y_train)

    return rf.best_estimator_, rf.best_score_, rf.best_params_


def evaluate(model, x_train, y_train, x_validation, y_validation):
    """Evaluate model on validationset
    :param model: Scikit-learn estimator
    :param x_train: trainset features
    :param y_train: trainset label
    :param x_validation: validationset features
    :param y_validation: validationset label
    :return model: Scikit-learn estimator, fitted on the whole trainset
    :return evaluations: prediction on validation set
    :return confusion_matrix: confusion matrix 
    :return accuracy_score: accuracy score 
    """ 

    # Refit the model on the whole train set
    rf = RandomForestClassifier(n_estimators = model['n_estimators'], random_state = 42, n_jobs = -1, warm_start = False)
    rf.fit(x_train, y_train) 

    # Evaluate on validation set
    evaluations = rf.predict(x_validation)

    # Metrics for evaluation
    confusion = confusion_matrix(y_validation, evaluations)
    accuracy = accuracy_score(y_validation, evaluations)

    return rf, evaluations, confusion, accuracy


def predict(model, x_test):
    """Predict values from the trained model
    :param model: Scikit-learn estimator
    :param x_validation: validationset features
    :return predictions: prediction on  set
    """

    # Evaluate on set
    predictions = model.predict(x_test)

    return predictions



## Model 1 (using extracted features of a molecule as input)
# Prepare the data
[x_train, y_train, x_validation, y_validation, x_test, y_test] = read_split_data(data_path = "dataset_single.csv", model_num = 1, train_ratio = 0.75, validation_ratio = 0.15, test_ratio = 0.10)

# Train the model
[estimator, score, params] = train(x_train, y_train)
print(params)

# Evaluate the model
[model, evaluations, confusion, accuracy] = evaluate(model = params, x_train = x_train, y_train = y_train, x_validation = x_validation, y_validation = y_validation)
print("Confusion matrix - Validation:", confusion)
print("Accuracy - Validation:", accuracy)


# Predict data from the model
predictions = predict(model, x_test)
print("Accuracy - Test:", accuracy_score(y_test, predictions))



## Model 2 (using the smile string character as input)
# Prepare the data
#[x_train, y_train, x_validation, y_validation, x_test, y_test] = read_split_data(data_path = "dataset_s.csv", model_num = 2, train_ratio = 0.75, validation_ratio = 0.15, test_ratio = 0.10)
