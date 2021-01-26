import tensorflow.keras as K
from NeuralNet import NeuralNet


class FFNet(NeuralNet):
    def __init__(self, X, y, topology, split=0.8, batch_size=64, epochs=200,
                 min_delta=0, patience=10, lr=0.0001, saveModel=False, models_dir='models'):
        self.topology = topology
        split = split
        self.batch_size = batch_size
        self.epochs = epochs
        min_delta = min_delta
        patience = patience
        self.lr = lr
        super(FFNet, self).__init__(X, y, epochs=epochs, split=split,
                                    batch_size=batch_size, min_delta=min_delta,
                                    patience=patience, saveModel=saveModel,
                                    models_dir=models_dir)

    def model(self):
        # encoder
        model = K.models.Sequential()
        # Hidden layers
        for i, nodes in enumerate(self.topology):
            if i == 0:
                model.add(K.layers.Dense(self.topology[1], input_dim=self.topology[0], activation='relu'))
            elif i < len(self.topology) - 2:
                model.add(K.layers.Dense(self.topology[i + 1], activation='relu'))
            else:
                # Last layer (output)
                model.add(K.layers.Dense(self.topology[i], activation='relu'))
        optimizer = K.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        model.summary()
        return model

