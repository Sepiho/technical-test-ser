#!/usr/bin/env python3
from pandas import read_csv
import servier.feature_extractor
#import feature_extractor
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from keras.models import Sequential
from keras.layers import Dense


def train(X_train, y_train, model_num):
    """Train with Grid-search cross validation to find the best hyperparameter for model 1 (random forest), and create model 2 (CNN)
    :param X_train: trainset features
    :param y_train: trainset label
    :param model_num: int, model of interest (1 or 2)
    :return model: model with the best parameter in a dict
    """


    # Model 1 - Random forest (to speed up processing, I comment the line searching for the best parameters with a lot of possibilities)
    if model_num == 1:
        # model = GridSearchCV(RandomForestClassifier(), cv = 8, param_grid = {"n_estimators": np.linspace(100, 2000, 200).astype('int'), "max_depth": [6, 8, 10]}, scoring = 'f1', n_jobs = -1)
        model = GridSearchCV(RandomForestClassifier(), cv = 8, param_grid = {"n_estimators": np.linspace(100, 1000, 200).astype('int'), "max_depth": [6]}, scoring = 'f1', n_jobs = -1)
        model.fit(X_train, y_train)

    # Model 2 - CNN
    elif model_num == 2:
        # Fix random seed for reproducibility
        np.random.seed(7)

        # Create the model 2 - CNN
        model = Sequential()
        model.add(Dense(15, input_dim = 4, activation = 'relu'))
        model.add(Dense(8, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # Binary crossentropy is a loss function that is used in binary classification tasks

        model.fit(X_train, y_train, epochs = 50, batch_size = 50, verbose = 0)

    else:
        return print("A model number must be choosen between 1 or 2")

    return model


def main():
    import sys
    train(sys.argv[1], sys.argv[2], sys.argv[3])


# Take inputs from command line, if launch that way
if __name__ == "__main__":
    main()