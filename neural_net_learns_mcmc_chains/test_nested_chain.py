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

# file = "/home/isidro/chains_server/owaCDM_phy_HD+SN+BBAO+Planck_nested_dynesty_multi_1.txt"
file = "/home/isidro/SimpleMC/chains/LCDM_phy_HD+SN+BBAO+Planck_nested_dynesty_multi_1.txt"

split = 0.8
epochs = 100
nlayers = 3
batch_size = 8
numNeurons = 50

data = np.loadtxt(file, usecols=(1,2,3,4,5,6))
params = data[7500:8000, 1:]
logL = data[7500:8000, 0]



print("len original data {}".format(len(data)))

# sparse = np.arange(0, 14000, 10)
# data_r = data[sparse]
# params = data_r[:, 1:]
# logL = data_r[:, 0]
nparams = len(params)

randomize = np.random.permutation(nparams)
params = params[randomize]
logL = logL[randomize]


# scaler = StandardScaler()
# scaler.fit(logL.reshape(-1,1))
# logL = scaler.transform(logL.reshape(-1,1))

ntrain = int(split * nparams)
indx = [ntrain]

params_training, _ = np.split(params, indx)
# params_training, params_testing = np.split(params, indx)
logL_training, _ = np.split(logL, indx)

nparams = len(params_training)
ntrain = int(split * nparams)
indx = [ntrain]
params_training, params_testing = np.split(params_training, indx)
logL_training, logL_testing = np.split(logL_training, indx)


# create scaler
scaler = StandardScaler()
# # scaler = MinMaxScaler(feature_range=(-1,1))
# # fit scaler on data
scaler.fit(params_training)
# # apply transform
params_training = scaler.transform(params_training)
params_testing = scaler.transform(params_testing)

# inverse transform
# inverse = scaler.inverse_transform(standardized)



print("Total len dataset {}," 
      "len training set {}, len test set {}".format(nparams, 
                                                 len(params_training), len(params_testing)))


callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                   min_delta=0,
                                   patience=10,
                                   restore_best_weights=True),
			tf.keras.callbacks.ReduceLROnPlateau(patience=5)
			]

n_cols = params_training.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(numNeurons, activation='relu', input_shape=(n_cols,)),
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(numNeurons, activation='relu'),
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(numNeurons, activation='relu'),
    # tf.keras.layers.Dropout(0.1),
    # tf.keras.layers.Dense(numNeurons, activation='relu', input_shape=(n_cols,)),
    tf.keras.layers.Dense(1, activation='linear')
    # 
    
])

model.compile(optimizer='adam', loss='mean_squared_error')

timei = time.time()
history = model.fit(params_training,
                    logL_training,
                    validation_data=(params_testing,
                                     logL_testing),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=callbacks)  

timef = (time.time() - timei)/60.

# u = uncertainty(history.history['loss'])
v = uncertainty(history.history['val_loss'])
# plt.plot(u)
plt.plot(v)
plt.xlabel('epoch')
plt.ylabel('uncertainty')
# plt.legend(['train', 'val'], loc='upper left')
plt.title('model uncertainty: {:.3}, {:.3} min'.format(np.squeeze(v.min()), timef))
plt.savefig("uncertainty_{}_layers_{}_nodes_{}_batchsize_10-4_xstandar".format(nlayers,numNeurons, batch_size))
plt.show()


plt.plot(history.history['loss'], label='training set')
plt.plot(history.history['val_loss'], label='validation set')
# plt.title('model loss function: {:.3} min'.format(timef))
plt.ylabel('loss function')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.savefig("loss_{}_layers_{}_nodes_{}_batchsize_10-4_xstandar".format(nlayers,numNeurons, batch_size))
plt.show()


# finalTestSetPoints = data[8000:, 1:]
# finalTestSetLikes = data[8000:, 0]
# # create scaler
# # scaler = StandardScaler()
# # scaler = MinMaxScaler(feature_range=(-1,1))
# # fit scaler on data
# # scaler.fit(params_training)
# # apply transform
# # finalTestSetPoints_scal = scaler.transform(finalTestSetPoints)

# predictions = model.predict(finalTestSetPoints)

# d = np.sqrt(np.sum(finalTestSetPoints**2, axis=1))

# plt.scatter(d, finalTestSetLikes, label="nested likes", c='r')

# plt.scatter(d, predictions, label="neural likes", c='g')
# plt.title('Likes bayesian vs neural likes')
# plt.ylabel('Likelihood value')
# plt.xlabel('$||v||$')
# plt.legend()

# plt.savefig("neuralLikes.png")
# plt.show()

# # finalTestSetPoints_scal = scaler.transform(finalTestSetPoints)

# predictions = model.predict(finalTestSetPoints)

# d = np.sqrt(np.sum(finalTestSetPoints**2, axis=1))

# plt.plot(finalTestSetLikes, label="nested likes", c='r')

# plt.plot(predictions, label="neural likes", c='g')
# plt.title('Likes bayesian vs neural likes')
# plt.ylabel('Likelihood value')
# plt.xlabel('index')
# plt.legend()

# plt.savefig("neuralLikes2.png")
# plt.show()


