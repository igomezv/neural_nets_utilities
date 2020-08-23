import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def uncertainty(loss):
    # test_loss = np.sqrt(loss)
    # return np.squeeze(test_loss.min())
    return np.sqrt(loss)

file = "/home/isidro/Desktop/chains_server/chains_1000/chains/owaCDM_phy_HD+SN+BBAO+Planck_nested_dynesty_multi_1.txt"

split = 0.8
numNeurons = 100
epochs = 200
nlayers = 1

params = np.loadtxt(file, usecols=(2,3,4,5,6))
logL = np.loadtxt(file, usecols=(1))

nparams = len(params)
randomize = np.random.permutation(nparams)
params = params[randomize]
logL = logL[randomize]

maxLogL = np.max(logL)
minLogL = np.min(logL)
ntrain = int(split * nparams)
indx = [ntrain]
params_training, params_testing = np.split(params, indx)
logL_training, logL_testing = np.split(logL, indx)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                   min_delta=0.001,
                                   patience=10,
                                   restore_best_weights=True)]

n_cols = params_training.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(numNeurons, activation='relu', input_shape=(n_cols,)),
    # tf.keras.layers.Dense(numNeurons, activation='relu'),
    # tf.keras.layers.Dense(numNeurons, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

ti = time.time()
history = model.fit(params_training,
                    logL_training,
                    validation_data=(params_testing,
                                     logL_testing),
                    epochs=epochs, batch_size=32,
                    callbacks=callbacks)  

tf = (time.time() - ti)/60.
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
u = uncertainty(history.history['loss'])
v = uncertainty(history.history['val_loss'])
plt.plot(u)
plt.plot(v)
plt.xlabel('epoch')
plt.ylabel('uncertainty')
plt.legend(['train', 'val'], loc='upper left')
plt.title('model uncertainty {}'.format(tf))
plt.savefig("uncertainty_{}_{}_layers_{}nodes".format(epochs, nlayers, numNeurons))
plt.show()


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss function')
# plt.ylabel('loss function')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig("loss_{}_{}_layers".format(epochs, nlayers))
# plt.show()
