#!/usr/bin/env python3

from pandas import read_csv
from servier.feature_extractor import fingerprint_features
#import feature_extractor
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from keras.models import Sequential
from keras.layers import Dense



def read_split_data(data_path, model_num, train_ratio, validation_ratio, test_ratio):
    """Split original data into train data, validation data, and test data.
    :param data_path: str, path to the a CSV data file
    :param model_num: int, model of interest (1 or 2)
    :param train_ratio: float, proportion of the original data for trainset, must be from 0 to 1
    :param validation_ratio: float, proportion of the original data for validationset, must be from 0 to 1
    :param test_ratio: float, proportion of the original data for testset, must be from 0 to 1
    :return X_train: X train dataset
    :return y_train: y train dataset
    :return X_validation: X validation dataset
    :return y_validation: y validation dataset
    :return X_test: X test dataset
    :return y_test: Y test dataset
    """


    # QC of he input ratios
    sum_ratios = train_ratio+validation_ratio+test_ratio
    if sum_ratios != 1:
        return print("Sum of the three ratios should be 1")
    
    # Read data, and preprocessing depending on the chosen model
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


def main():
    import sys
    read_split_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


# Take inputs from command line, if launch that way
if __name__ == "__main__":
    main()
