# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 22:12:29 2021

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
from rdkit import Chem
from rdkit.Chem import Descriptors
from keras.models import Sequential
from keras.layers import Dense


## Functions
def read_split_data(data_path, model_num, train_ratio, validation_ratio, test_ratio):
    """Split original data into train data validation data, and test data.
    :param data_path: str, path to the a CSV data file
    :param model_num: int, model of interest (1 or 2)
    :param train_ratio: float, proportion of the original data for trainset, must be from 0 to 1
    :param validation_ratio: float, proportion of the original data for validationset, must be from 0 to 1
    :param test_ratio: float, proportion of the original data for testset, must be from 0 to 1
    :return X_train:
    :return y_train:
    :return X_validation:
    :return y_validation:
    :return X_test:
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
		# Extract 4 descriptors of the molecule
    	X = dataset
    	X['mol'] = X['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    	X['tpsa'] = X['mol'].apply(lambda x: Descriptors.TPSA(x))
    	X['mol_w'] = X['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    	X['num_valence_electrons'] = X['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    	X['num_heteroatoms'] = X['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    	X = X.drop(columns=['smiles', 'mol', 'P1', 'mol_id'])

    else:
    	return print("A model number must be choosen between 1 or 2")

    # We first split the dataset to create the train subset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - train_ratio, random_state=1)
    
    # We split the test dataset into a final test subset and validation subset
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), random_state = 1, shuffle = False)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def train(X_train, y_train, model_num):
    """Train with Grid-search cross validation to find the best hyperparameter
    :param X_train: trainset features
    :param y_train: trainset label
    :param model_num: int, model of interest (1 or 2)
    :return model: model with the best hyper parameter in a dict
    """

    if model_num == 1:
    	model = GridSearchCV(RandomForestClassifier(), cv = 8, param_grid = {"n_estimators": np.linspace(200, 3000, 200).astype('int'), "max_depth": [8, 9, 10, 11, 12, 20]}, scoring = 'f1', n_jobs = -1)
    	model.fit(X_train, y_train)

    elif model_num == 2:
    	# Fix random seed for reproducibility
    	np.random.seed(7)

    	# Create the model
    	model = Sequential()
    	model.add(Dense(15, input_dim = 4, activation = 'relu'))
    	model.add(Dense(8, activation = 'relu'))
    	model.add(Dense(1, activation = 'sigmoid'))
    	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    	model.fit(X_train, y_train, epochs = 5, batch_size = 32, verbose = 0)

    else:
    	return print("A model number must be choosen between 1 or 2")

    return model


def evaluate(model, X_train, y_train, X_validation, y_validation, model_num):
    """Evaluate model on validationset
    :param model: model
    :param X_train: trainset features
    :param y_train: trainset label
    :param X_validation: validationset features
    :param y_validation: validationset label
    :param model_num: int, model of interest (1 or 2)
    :return model: model, fitted on the whole trainset
    :return evaluations: prediction on validation set
    :return confusion_matrix: confusion matrix 
    :return accuracy_score: accuracy score 
    """ 

    if model_num == 1:
    	# Refit the model on the whole train set
    	model = RandomForestClassifier(n_estimators = model['n_estimators'], max_depth = model['max_depth'], random_state = 6, n_jobs = -1, warm_start = False)
    	model.fit(X_train, y_train)

    	# Evaluate on validation set
    	evaluations = model.predict(X_validation)

    	# Metrics for evaluation
    	confusion_loss = confusion_matrix(y_validation, evaluations)
    	accuracy = accuracy_score(y_validation, evaluations)

    elif model_num == 2:
    	# Metrics for evaluation
    	confusion_loss, accuracy = model.evaluate(X_validation, y_validation)
    	evaluations = model.predict(X_validation)

    else:
    	return print("A model number must be choosen between 1 or 2")

    return model, evaluations, confusion_loss, accuracy


def predict(model, X_test):
    """Predict values from the trained model
    :param model: model
    :param X_test: testset features
    :return predictions: prediction values on set
    """

    # Evaluate on set
    predictions = model.predict(X_test)

    return predictions









# ## Model 1 (using extracted features of a molecule as input) - Random forest
# # Prepare the data
# [X_train, y_train, X_validation, y_validation, X_test, y_test] = read_split_data(data_path = "data/dataset_single.csv", model_num = 1, train_ratio = 0.75, validation_ratio = 0.15, test_ratio = 0.10)

# # Train the model
# model = train(X_train, y_train, model_num = 1)
# print(model.best_params_)

# # Evaluate the model
# [model, evaluations, confusion, accuracy] = evaluate(model = model.best_params_, X_train = X_train, y_train = y_train, X_validation = X_validation, y_validation = y_validation, model_num = 1)
# print("Confusion matrix - Validation:", confusion)
# print("Accuracy - Validation:", accuracy)


# # Predict data from the model
# predictions = predict(model, X_test)
# print("Accuracy - Test:", accuracy_score(y_test, predictions))



## Model 2 (using the smile string character as input) - LSTM
# Prepare the data
[X_train, y_train, X_validation, y_validation, X_test, y_test] = read_split_data(data_path = "data/dataset_single.csv", model_num = 2, train_ratio = 0.75, validation_ratio = 0.15, test_ratio = 0.10)

# Train the model
model = train(X_train, y_train, model_num = 2)
# print(model.summary())

# Evaluate the model
[model, evaluations, loss, accuracy] = evaluate(model = model, X_train = X_train, y_train = y_train, X_validation = X_validation, y_validation = y_validation, model_num = 2)
print("Loss- Validation:", loss)
print("Accuracy - Validation:", accuracy)

# Predict data from the model
predictions = predict(model, X_test)
