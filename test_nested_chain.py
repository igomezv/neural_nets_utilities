import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# MinMaxScaler(feature_range=(-1,1))
import time

def uncertainty(loss):
    # test_loss = np.sqrt(loss)
    # return np.squeeze(test_loss.min())
    return np.sqrt(loss)

file = "/home/isidro/chains_server/owaCDM_phy_HD+SN+BBAO+Planck_nested_dynesty_multi_1.txt"

split = 0.8
numNeurons = 200
epochs = 100
nlayers = 3
batch_size = 64

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

# params_testing = params_testing[:200,:]
# create scaler
scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(-1,1))
# fit scaler on data
scaler.fit(params_training)
# apply transform
params_training = scaler.transform(params_training)
params_testing = scaler.transform(params_testing)

# inverse transform
# inverse = scaler.inverse_transform(standardized)
print(logL_testing.shape, type(logL_testing))

print("Total len dataset {}," 
      "len training set {}, len test set {}".format(nparams, 
                                                 len(params_training), len(params_testing)))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                   min_delta=1e-4,
                                   patience=2,
                                   restore_best_weights=True)]

n_cols = params_training.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(numNeurons, activation='relu', input_shape=(n_cols,)),
    tf.keras.layers.Dense(numNeurons, activation='relu'),
    tf.keras.layers.Dense(numNeurons, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

ti = time.time()
history = model.fit(params_training,
                    logL_training,
                    validation_data=(params_testing,
                                     logL_testing),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=callbacks)  

tf = (time.time() - ti)/60.

# u = uncertainty(history.history['loss'])
# v = uncertainty(history.history['val_loss'])
# plt.plot(u)
# plt.plot(v)
# plt.xlabel('epoch')
# plt.ylabel('uncertainty')
# plt.legend(['train', 'val'], loc='upper left')
# plt.title('model uncertainty: {:.3}, {:.3} min'.format(np.squeeze(u.min()), tf))
# plt.savefig("uncertainty_{}_{}_layers_{}nodes".format(epochs, nlayers, numNeurons))
# plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss function: {:.3} min'.format(tf))
plt.ylabel('loss function')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.savefig("loss_{}_{}_layers_{}_nodes_{}_batchsize_10-4_xstandar".format(epochs, nlayers,numNeurons, batch_size))
plt.show()
