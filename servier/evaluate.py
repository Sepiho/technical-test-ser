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

def evaluate(model, X_train, y_train, X_validation, y_validation, model_num):
    """Evaluate model 1 or 2 on validationset
    :param model: model
    :param X_train: trainset features
    :param y_train: trainset label
    :param X_validation: validationset features
    :param y_validation: validationset label
    :param model_num: int, model of interest (1 or 2)
    :return model: model, fitted on the whole trainset
    :return evaluations: prediction on validation set
    :return confusion_loss: confusion matrix (model 1), or loss (model 2)
    :return accuracy: accuracy score 
    """


    # Model 1 - Random forest
    if model_num == 1:
        # Refit the model on the whole train set
        model = RandomForestClassifier(n_estimators = model['n_estimators'], max_depth = model['max_depth'], random_state = 6, n_jobs = -1, warm_start = False)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        evaluations = model.predict(X_validation)

        # Metrics for evaluation
        confusion_loss = confusion_matrix(y_validation, evaluations)
        accuracy = accuracy_score(y_validation, evaluations)

    # Model 2 - CNN
    elif model_num == 2:
        # Metrics for evaluation
        confusion_loss, accuracy = model.evaluate(X_validation, y_validation)
        evaluations = model.predict(X_validation)

    else:
        return print("A model number must be choosen between 1 or 2")

    return model, evaluations, confusion_loss, accuracy


def main():
    import sys
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])


# Take inputs from command line, if launch that way
if __name__ == "__main__":
    main()