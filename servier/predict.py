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

def predict(model, X_test):
    """Predict values from the trained model
    :param model: model
    :param X_test: testset features
    :return predictions: prediction values on set
    """

    # Evaluate on set
    predictions = model.predict(X_test)

    return predictions


def main():
    import sys
    predict(sys.argv[1], sys.argv[2])


# Take inputs from command line, if launch that way
if __name__ == "__main__":
    main()