from OptInterpolation import *
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.regularizers import l2
from Dataset import *
from utils import *

"""
Abstract Regressor class.
"""


class Regressor:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, epochs=20, verbose=True, plot=False, patience=5):
        """
        Method responsible for the training of the model
        :param epochs: num of training epochs
        :param verbose: verbose training output
        :param plot: plot training curve
        :param patience: patience for the early stopping
        :return: training history
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        self.history = self.model.fit(self.df.train_gen, epochs=epochs, validation_data=self.df.val_gen,
                                      callbacks=[callback],
                                      verbose=verbose)
        if plot:
            plt.plot(self.history.history['loss'], label='training loss')
            plt.plot(self.history.history['val_loss'], label='validation loss')
            plt.legend()
            plt.show()

        return self.history

    def predict(self, *args, **kwargs):
        pass

    def create_model(self, *args, **kwargs):
        pass

    def save_model(self, name):
        """
        :param name: name for the saved model
        """
        self.model.save(name)
        print('Model has been saved with a name ', name, '\n')

        return
