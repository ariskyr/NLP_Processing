import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import Preprocess as pp

#tf config
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
batch_size = 16
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False)
no_classes = 20
no_epochs = 200
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6)
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
    inputs = keras.Input(shape=(8520,))
    x = layers.Dense(40,activation='relu', kernel_regularizer=l2(0.5))(inputs)
    x = layers.Dense(40,activation='relu', kernel_regularizer=l2(0.5))(x)
    outputs = layers.Dense(no_classes,activation='sigmoid')(x)
    model = keras.Model(inputs=inputs,outputs=outputs,name='DeliciousMIL_model')

    print(model.summary())
    #compile model
    model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy', 'mse'],
                )
    
    print('-----------------------------------------------------------------------------')
    print(f'Training for fold {fold} ...')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    mc = ModelCheckpoint('modelA4a.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
    
    history = model.fit(X_data[train], Y_data[train],
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity,
                        validation_data=(X_data[test], Y_data[test]),
                        callbacks=[es,mc])
    
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

# == Plot model history accurracy ==
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# == Plot model history CE loss ==
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# == Plot model history MSE ==
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


x= 0