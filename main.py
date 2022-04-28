from pyclbr import Function
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import Preprocess as pp

#tf config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#########
#for the full preprocessing:
#########
#rawSplitSet,centeredSplitSet,NormalizedSplitSet, StandardizedSplitSet, CNSplitSet = pp.preprocessData()

#########
# if you dont wanna wait for the preprocessing of data
#########
#rawData = pp.decompress_pickle("SavedArrays\\rawSplitSet.pbz2")
centeredData = pp.decompress_pickle("SavedArrays\\centeredSplitSet.pbz2")
#normalizedData = pp.decompress_pickle("SavedArrays\\normalizedSplitSet.pbz2")
#standardizedData = pp.decompress_pickle("SavedArrays\\standardizedSplitSet.pbz2")
#CNData = pp.decompress_pickle("SavedArrays\\CNSplitsSet.pbz2")

X_data = centeredData['X_train'].to_numpy()
Y_data = centeredData['Y_train'].to_numpy()
X_data = np.stack(X_data)
Y_data = np.stack(Y_data)

# Model configuration
batch_size = 32
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False)
no_classes = 20
no_epochs = 100
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
verbosity = 2
num_folds = 5

#Metrics containers per fold
acc_per_fold = []
loss_per_fold = []
mse_per_fold = []

#K-fold Cross Validator from scikit
kfold = KFold(n_splits=num_folds, shuffle=True)

#Model
fold = 1
for train,test in kfold.split(X_data, Y_data):
    #architecture
    inputs = keras.Input(shape=(8520))
    dense1 = layers.Dense(20,activation='relu')(inputs)
    outputs = layers.Dense(no_classes,activation='sigmoid')(dense1)
    model = keras.Model(inputs=inputs,outputs=outputs,name='DeliciousMIL_model')

    #compile model
    model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy','mse'],
                )
    
    print('-----------------------------------------------------------------------------')
    print(f'Training for fold {fold} ...')

    history = model.fit(X_data[train], Y_data[train],
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity)

    scores = model.evaluate(X_data[test], Y_data[test], verbose=2)
    print(f'Score for fold {fold}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%; {model.metrics_names[2]} of {scores[2]}.')
    loss_per_fold.append(scores[0])
    acc_per_fold.append(scores[1]*100)
    mse_per_fold.append(scores[2])

    fold += 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - MSE: {mse_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print(f'> MSE: {np.mean(mse_per_fold)}')
print('------------------------------------------------------------------------')
x= 0