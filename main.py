# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Sat Feb 6 22:12:29 2021

@author: Sophie Sebille

This file launchs all the functions of the package with an example.
"""

## Load libraries and functions
import os
import argparse, sys
from sklearn.metrics import accuracy_score
from servier.read_split_data import read_split_data
from servier.train import train
from servier.evaluate import evaluate
from servier.predict import predict



def main(argv):
   ## Variables
   parser = argparse.ArgumentParser(description='')
   parser.add_argument("--datapath", required=True, help='Path to the data')
   parser.add_argument("--model_num", required=True, type=int, default=2, help='Number of the model (1 or 2)')
   parser.add_argument("--train_ratio", type=float, default=0.75, help='Ratio for the train dataset')
   parser.add_argument("--validation_ratio", type=float, default=0.15, help='Ratio for the validation dataset')
   parser.add_argument("--test_ratio", type=float, default=0.10, help='Ratio for the test dataset')
   args = parser.parse_args()

   datapath = args.datapath
   model_num = args.model_num
   train_ratio = args.train_ratio
   validation_ratio = args.validation_ratio
   test_ratio = args.test_ratio


   ## Run the model
   # Prepare the data
   [X_train, y_train, X_validation, y_validation, X_test, y_test] = read_split_data(datapath, model_num, train_ratio, validation_ratio, test_ratio)

   # Train the model
   model = train(X_train, y_train, model_num)

   # Evaluate the model
   if model_num == 1:
      model = model.best_params_
   [model, evaluations, confusion_loss, accuracy] = evaluate(model, X_train, y_train, X_validation, y_validation, model_num)
   if model_num == 1:
      print("\nConfusion matrix metric - Validation set:", confusion_loss)
   elif model_num == 2:
      print("\nLoss metric - Validation set:", confusion_loss)
   print("Accuracy metric - Validation set:", accuracy)

   # Predict data from the model
   predictions = predict(model, X_test)
   if model_num == 2:
      predictions[predictions < 0.5] = 0
      predictions[predictions >= 0.5] = 1
   print("\nAccuracy metric - Test set:", accuracy_score(y_test, predictions))


if __name__ == "__main__":
   main(sys.argv[1:])




