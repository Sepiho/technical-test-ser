# Technical test from S
The prediction of a drug molecule properties plays an important role in the drug design process. The molecule properties are the cause of failure for 60% of all drugs in the clinical phases. A multi parameters optimization using machine learning methods can be used to choose an optimized molecule to be subjected to more extensive studies and to avoid any clinical phase failure.\
This python module allows the user to train the model, evaluate the model, and predict the property P1 for any given molecule smile.
\
\

## Models
### Model 1 (using extracted features of a molecule as input) - Random forest
This python module contains a random forest model that takes the extracted features of a molecule as input and predict the P1 property. Best hyperparameters for this model are find via Grid-search cross validation. The random forest algortihm was chosen as it can be used for classifications, typically provides high accuracy, will handle the missing values and maintain the accuracy of a large proportion of data, if there are more trees, it usually won’t allow overfitting trees in the model, and it has the power to handle a large data set.
\
\

### Model 2 (using the smile string character as input) - Convolutional neural network (CNN)
The second model, CNN, takes the smile string character as input and predict the P1 property. Inputs fed to the CNN model are characteristics of the molecule calculated by the script: topological polar surface area (TPSA), exact molecular weight, number of radical electrons, and  number of heteroatoms.\
Next is the architecture of the chosen model:\
Model: "sequential"\
Layer (type)                 Output Shape              Param #\
dense (Dense)                (None, 15)                75\
dense_1 (Dense)              (None, 8)                 128\
dense_2 (Dense)              (None, 1)                 9\
\
Total params: 212\
Trainable params: 212\
Non-trainable params: 0\
\
\

## Requirements
Before you begin, ensure you have met the following requirements:
* You have installed the latest version of Python 3.7
* You have a Windows, Linux, or Mac machine
* You have installed rdkit, if you don't plan on using docker to run the program. See https://www.rdkit.org/docs/Install.html for instructions
* You have installed docker, if you plan to use it to run the program. See https://docs.docker.com/get-docker/ for instructions
* You have read the documentation
\
\

## Installation
### Via setup.py (if you don't plan on using docker to run the scripts/functions)
To install, follow these steps:
```
python setup.py install
```
\


### Via docker (with an already dowloaded image container_servier.tar)
To install via Docker, follow these steps:
```
docker load -i container_servier.tar
```
\


### Via docker (by building the docker image)
To install via Docker, follow these steps:
```
docker build . -t servier
```
\


## Example
### Via API
A flask API (with one route /predict) exists and the client can send a molecule smile and get the P1 prediction property of the molecule based on model 2 - CNN. 

To do this, follow these steps, from the server side:
```
cd api
python flask-api.py
```

Then, the client can send their molecule smile with a similar line in a terminal (jut change the CCc1cccc2c1NC(=O)C21C2C(=O)N(Cc3ccccc3)C(=O)C2C2CCCN21 smile to the smile of interest):
```
curl -X POST localhost:5000/predict?smiles="CCc1cccc2c1NC(=O)C21C2C(=O)N(Cc3ccccc3)C(=O)C2C2CCCN21"
```
\


### Via command line in a bash console
```
python main.py --datapath="/data/dataset_single.csv" --model_num=2 --train_ratio=0.75 --validation_ratio=0.15 --test_ratio=0.10
```
\


### Via docker (the name of the container is servier)
```
docker run -v /home/sepiho/technical-test-serv/data/dataset_single.csv:/app/data.csv servier --datapath="data.csv" --model_num=2 --train_ratio=0.75 --validation_ratio=0.15 --test_ratio=0.10
```
\


## Troubleshooting

If you want to contact me you can reach me at <sophie.sebille@laposte.net>.
\

## License

This project uses the following license: MIT.
