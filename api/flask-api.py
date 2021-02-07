from flask import Flask, request
import pandas as pd
from keras.models import load_model
from rdkit import Chem
from rdkit.Chem import Descriptors

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    var_send = request.args.get('smiles')
    smiles = {'smiles' : [var_send]}
    X = pd.DataFrame(smiles, columns = ['smiles'])
    X['mol'] = X['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    X['tpsa'] = X['mol'].apply(lambda x: Descriptors.TPSA(x))
    X['mol_w'] = X['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    X['num_valence_electrons'] = X['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    X['num_heteroatoms'] = X['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    X = X.drop(columns = ['smiles', 'mol'])

    # Load the model2 trained before
    model2 = load_model('my_model2.h5')

    # Predict the outcome
    pred = model2.predict(X)
    print(pred)

    # Transform probability into categorical value
    if pred > 0.5:
    	prediction = '1'
    else:
    	prediction = '0'

    return "P1: " + prediction + "\n"


if __name__ == '__main__':
    app.run(debug=True)