import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


class NeuralNet(object):
    def __init__(self, X, y, epochs=200, split=0.8, batch_size=64, min_delta=0, patience=10,
                 saveModel=False, models_dir='models'):

        split = split
        self.batch_size = batch_size
        self.epochs = epochs
        min_delta = min_delta
        patience = patience

        self.saveModel = saveModel
        models_dir = models_dir

        randomize = np.random.permutation(len(X))
        X = X[randomize]
        y = y[randomize]
        ntrain = int(split * len(X))
        indx = [ntrain]
        self.X_train, self.X_test = np.split(X, indx,  axis=0)
        self.y_train, self.y_test = np.split(y, indx,  axis=0)
        self.model = self.model()

        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           min_delta=min_delta,
                                                           patience=patience,
                                                           restore_best_weights=True)]

        if self.saveModel:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                                  os.path.join(models_dir, 'epoch_{epoch:02d}_mse_{val_loss:.4f}.h5'),
                                  monitor='val_loss', save_weights_only=False, save_best_only=False))

        self.trained_model = self.fit()

    def model(self):
        err = "NeuralNet: You need to implement an model"
        raise NotImplementedError(err)

    def fit(self):
        return self.model.fit(self.X_train, self.y_train,
                              batch_size=self.batch_size,
                              epochs=self.epochs, verbose=1,
                              validation_data=(self.X_test, self.y_test),
                              callbacks=self.callbacks)

    def predict(self, new_vals=None):
        if new_vals is None:
            return self.model.predict(self.y_test)
        else:
            return self.model.predict(new_vals)

    def plot(self, **kwargs):
        plt.clf()
        outputname = kwargs.pop('outputname', 'loss')
        train_color = kwargs.pop('train_color', 'r')
        val_color = kwargs.pop('val_color', 'g')
        show = kwargs.pop('show', False)
        title = kwargs.pop('title', 'Loss function')

        plt.plot(self.trained_model.history['loss'], color=train_color)
        plt.plot(self.trained_model.history['val_loss'], color=val_color)
        plt.title(title)
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('{}.png'.format(outputname))
        if show:
            plt.show()




