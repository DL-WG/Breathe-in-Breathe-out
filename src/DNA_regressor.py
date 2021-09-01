from OptInterpolation import *
from sklearn.metrics import r2_score
from LSTM_regressor import *
from Regressor import *
import tensorflow as tf

"""
Dynamic Neural Assimilation implementation.
Implements Regressor abstract class.
"""


class DNA_regressor(Regressor):
    def __init__(self, df, n_units=[], lr=0.001, inp_shape=5, reg_param=0.0001, model=None, alfa=0.5):
        """
        Object can be initialised with model configuration or with pretrained model
        :param df: Dataset object with all the train, val and test data
        :param model: pretrained model
        :other param:  model configuration if new model needs to be trained
        """
        self.df = df
        self.alfa = alfa

        if model is None:
            self.model = self.create_model(n_units, lr, inp_shape, reg_param)
        else:
            self.model = model

    def dna_loss(self, y_true, y_pred):
        """
        Custom DNA loss function.
        'alfa' controls how much we can trust observations from 0.0 to 1.0.
        :param y_true: ground truth values
        :param y_pred: predicted values
        :return: loss
        """
        squared_difference = tf.square(y_true - y_pred)
        x = tf.constant([1 - self.alfa, self.alfa], dtype=tf.float32)
        squared_difference = tf.multiply(x, squared_difference)

        return tf.reduce_mean(squared_difference, axis=-1)

    def create_model(self, n_units, lr, inp_shape, reg_param):
        """
        Method responsible for the creation of the model
        :return: LSTM model
        """
        model = Sequential()

        model.add(
            LSTM(n_units[0], return_sequences=True, input_shape=(None, inp_shape), kernel_regularizer=l2(reg_param)))
        for i in range(len(n_units[1:-1])):
            model.add(LSTM(n_units[i], return_sequences=True))
        model.add(LSTM(n_units[-1], return_sequences=False))
        model.add(Dense(1))

        model.compile(loss=self.dna_loss, optimizer=Adam(learning_rate=lr))
        # model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

        model.summary()

        return model

    def predict(self, plot=False, plot_length=120):
        """
        Method responsible for the assimilation of the test data
        :param plot_length: length of assimilated vs observed vs AMS-MINNI modelled graph
        :return: model assimilated values
        """
        self.preds = self.model.predict(self.df.test_gen)

        self.obs = pd.Series(
            self.df.y_scaler.inverse_transform(np.squeeze(self.df.test_data_out['obs'].values[self.df.n_input:])),
            index=self.df.test_idx)
        self.sim = pd.Series(
            self.df.y_scaler.inverse_transform(np.squeeze(self.df.test_data_out['sim'].values[self.df.n_input:])),
            index=self.df.test_idx)
        self.preds = pd.Series(self.df.y_scaler.inverse_transform(np.squeeze(self.preds)), index=self.df.test_idx)

        if plot:
            plt.plot(self.preds[360:360+min(plot_length, len(self.preds))], 'b--', label='DNA predicted')
            plt.plot(self.obs[360:360+min(plot_length, len(self.obs))], 'r-', label='Observations')
            plt.plot(self.sim[360:360+min(plot_length, len(self.sim))], 'y-.', label='AMS-MINNI modelled')
            plt.legend()
            plt.show()

        return self.preds  # , base_mse, test_mse

    def dna_metrics(self):
        """
        Method for the calculation of mean squared assimilation and
        mean squared forecasting error with respect to the observed values
        :return: MSE^F, MSE^A
        """
        sim = self.sim.values
        obs = self.obs.values
        preds = self.preds.values

        mse_f = np.sqrt(np.square(sim - obs).sum()) / np.sqrt(np.square(obs).sum())
        mse_a = np.sqrt(np.square(preds - obs).sum()) / np.sqrt(np.square(obs).sum())

        return mse_f, mse_a
