import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from tensorflow.keras.activations import swish

import numpy as np
import json
from datetime import datetime

import golois
import gc

class GOelanModel:
    
    def __init__(self, params):
        
        self.params = params
        self.planes = 31
        self.moves = 361

    def create_model(self):
        '''
        Creates keras model
        '''

        input = keras.Input(shape=(19, 19, self.planes), name='board')
        x = layers.Conv2D(self.params['filters'], 1, activation='relu', padding='same')(input)
        for i in range (10):
            x1 = layers.Conv2D(self.params['filters'], 3, activation='relu', padding='same')(x)
            x1 = layers.Conv2D(self.params['filters'], 3, padding='same')(x1)
            x = layers.add([x1,x])
            x = layers.ReLU()(x)
            x = layers.BatchNormalization()(x)
        policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
        policy_head = layers.Flatten()(policy_head)
        policy_head = layers.Activation('softmax', name='policy')(policy_head)
        value_head = layers.GlobalAveragePooling2D()(x)
        value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        # Build the model
        self.model = keras.Model(inputs=input, outputs=[policy_head, value_head])



    def load_model(self, model):
        assert model.split(".")[-1] == "h5", "Model argument is not a h5 file !"
        model_name = f"{self.params['drive_path']}/models/{model}"
        self.model = keras.models.load_model(model_name)

    def summary(self):
        return self.model.summary()

    def train(self):
        '''
        Compiles and creates training loop for GOelan Model
        '''
        N = self.params['N']
        today_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        total_history = []  # Collect all history

        input_data = np.random.randint(2, size=(N, 19, 19, self.planes))
        input_data = input_data.astype ('float32')

        policy = np.random.randint(self.moves, size=(N,))
        policy = keras.utils.to_categorical (policy)

        value = np.random.randint(2, size=(N,))
        value = value.astype ('float32')

        end = np.random.randint(2, size=(N, 19, 19, 2))
        end = end.astype ('float32')

        groups = np.zeros((N, 19, 19, 1))
        groups = groups.astype ('float32')

        print ("getValidation", flush = True)
        golois.getValidation (input_data, policy, value, end)
    	
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate = self.params['learning_rate'], momentum = 0.9),
                            loss = {'policy' : 'categorical_crossentropy', 'value' : 'binary_crossentropy'},
                            loss_weights = {'policy' : self.params['policy_loss_weight'], 'value' : self.params['value_loss_weight']},
                            metrics = {'policy': 'categorical_accuracy', 'value': 'mse'})

        for i in range(1, self.params['epochs'] + 1):
            print(f'----- Epoch {i} -----')
            golois.getBatch(input_data, policy, value, end, groups, i * N)

            print('\tTRAIN')
            history = self.model.fit(input_data,
                            {'policy': policy, 'value': value}, 
                            epochs = 1, batch_size = self.params['batch_size'])
            
            total_history.append(history.history)
            
            if (i % 5 == 0):
                gc.collect()

            # Validation
            print('\t VALIDATION')
            golois.getValidation(input_data, policy, value, end)
            val = self.model.evaluate(input_data,
                                [policy, value], verbose = 0, batch_size = self.params['batch_size'])
            
            # Save
            model_name = f'{today_date}_GOelan_{val[3]:.3f}_pol-acc_{val[4]:.3f}_val-mse_{i}_epochs'
            history_name = 'hist_'+model_name+'.json'
            self.model.save(f"{self.params['drive_path']}/models/{model_name}.h5")
            json.dump(total_history, open(f"{self.params['drive_path']}/history/{history_name}", 'w'))

            if (i % 5 == 0):
              print(f"\n[EPOCH {i}] Validation metrics \nValidation loss = {val[0]}\nPolicy loss = {val[1]}\nValue loss = {val[2]}\nPolicy accuracy = {val[3]}\nValue accuracy = {val[4]}\n")

        return history