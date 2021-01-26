from NeuralNet import NeuralNet
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, UpSampling2D, \
    MaxPooling2D, Flatten, Reshape, Conv2DTranspose, Dropout, ZeroPadding2D
from tensorflow.keras import Input

class ConvolAE(NeuralNet):
    def __init__(self, X, y, epochs=100, split=0.8, batch_size=128, min_delta=0,
                 patience=10, lr=0.0001, size=1048, saveModel=False, models_dir='models'):

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.latent_dim = size
        self.input_size = size
        self.e_model = self.encoder()
        self.d_model = self.decoder()
        super(ConvolAE, self).__init__(X, y, epochs=epochs, split=split,
                                       batch_size=batch_size, min_delta=min_delta,
                                       patience=patience, saveModel=saveModel,
                                       models_dir=models_dir)

    def model(self):
        input_cov = Input(shape=(self.input_size, self.input_size, 1))
        model = tf.keras.Model(input_cov, self.d_model(self.e_model(input_cov)),
                               name='autoencoder')
        # optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        # optimizer = tf.keras.optimizers.Adam(lr=0.00008, beta_1=0.9)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        # model.summary()

        return model

    def encoder(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(6, (3, 3), activation='relu', padding='same', input_shape=(self.input_size, self.input_size, 1)))
        # out = 1048x1048x5
        model.add(MaxPooling2D((2, 2), padding='same'))
        # out = 524x524x5
        model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
        # out = 524 x 524 x 5
        model.add(MaxPooling2D((2, 2), padding='same'))
        # out = 262 x 262 x 5
        model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
        # out = 262 x 262 x 2
        model.add(MaxPooling2D((2, 2), padding='same'))
        # out = 131 x 131 x 2
        model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
        # # out = 131 x 131 x 1
        model.add(Flatten()) # 131x131x1
        # latent dimension
        model.add(Dense(self.latent_dim))
        model.summary()
        # tf.keras.utils.plot_model(model, show_shapes=True)

        return model

    def decoder(self):
        model = tf.keras.Sequential()
        # model.add(Dense(131 * 131 * 2, input_shape=(self.latent_dim,)))
        # model.add(Reshape((131, 131, 2)))
        # # output = 131 x 131 x 1
        model.add(Dense(131 * 131 * 1, input_shape=(self.latent_dim,)))
        model.add(Reshape((131, 131, 1)))

        model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
        # output = 131 x 131 x 2
        model.add(UpSampling2D((2, 2)))
        # output = 262 x 262 x 2
        model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
        # output = 262 x 262 x 2
        model.add(UpSampling2D((2, 2)))
        # output = 524 x 524 x 2
        model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
        # output = 524 x 524 x 2
        model.add(UpSampling2D((2, 2)))
        # output = 1048 x 1048 x 2
        model.add(Conv2D(6, (3, 3), activation='relu', padding='same'))
        # output = 1048 x 1048 x 2
        model.add(Dropout(0.5))
        # output = 1048 x 1048 x 2
        model.add(Conv2D(1, (3, 3), activation='linear', padding='same'))
        # output = 1048 x 1048 x 1
        model.summary()
        # tf.keras.utils.plot_model(model, show_shapes=True)

        return model
