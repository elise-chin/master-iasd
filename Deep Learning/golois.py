import json
import argparse
import os
import tensorflow as tf
from datetime import datetime
from model import GOelanModel
import keras
from tensorflow.keras.utils import plot_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default=None, nargs='?')

if __name__ == '__main__':
    today_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Read parameters
    json_file = open("params.json")
    params = json.load(json_file)
    json_file.close() 

    args = parser.parse_args()
    model_name = args.model_name

    # Create Keras Model
    model = GOelanModel(params)

    # Instantiate GOelan Model
    if model_name:
      model.load_model(model_name)
    else:
      model.create_model()

    model.summary()
    #model.build(input_shape=(19, 19, 31))
    #plot_model(model, show_shapes=True, show_layer_names=True)

    # Train model
    history = model.train()

    # Results
    #pol_acc = history.history['policy_categorical_accuracy'][-1]
    #val_mse = history.history['value_mse'][-1]

    # Save model
    #model_name = f'{today_date}_GOelan_{pol_acc:.3f}_pol-acc_{val_mse:.3f}_val-mse_{params["N"]}_samples_{params["epochs"]}_epochs.h5'
    #model.model.save(f"{params['drive_path']}/models/{model_name}")
    #print("\n", model_name)