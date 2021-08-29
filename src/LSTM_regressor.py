from OptInterpolation import *
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.regularizers import l2
from Regressor import *

"""
Implementation of the LSTM based
model that replicates
the AMS-MINNI model forecast.
Implements Regressor abstract class.
"""


class LSTM_regressor(Regressor):

    def __init__(self, df, n_units=[], lr=0.001, inp_shape=5, reg_param=0.0001, model=None):
        """
        Object can be initialised with model configuration or with pretrained model
        :param df: Dataset object with all the train, val and test data
        :param model: pretrained model
        :other param:  model configuration if new model needs to be trained
        """
        self.df = df

        if model is None:
            self.model = self.create_model(n_units, lr, inp_shape, reg_param)
        else:
            self.model = model

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

        # model.compile(loss=self.dna_loss, optimizer=Adam(learning_rate=lr))
        model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

        model.summary()

        return model

    def predict(self, plot_length=120, show_mse=False, labels=['LSTM predicted', 'AMS-MINNI modelled']):
        """
        Method responsible for the prediction of the test data
        :param plot_length: length of predicted vs AMS-MINNI modelled graph
        :param show_mse: print mse and baseline mse
        :param labels: graph axis names
        :return: model predictions
        """
        preds = self.model.predict(self.df.test_gen)
        self.sim = pd.Series(
            self.df.y_scaler.inverse_transform(np.squeeze(self.df.test_data_out['sim'].values[self.df.n_input:])),
            index=self.df.test_idx)
        self.preds = pd.Series(self.df.y_scaler.inverse_transform(np.squeeze(preds)), index=self.df.test_idx)

        plt.plot(self.preds[:min(plot_length, len(self.preds))], 'b--', label=labels[0])
        plt.plot(self.sim[:min(plot_length, len(self.sim))], 'y-.', label=labels[1])
        plt.legend()
        plt.rcParams["figure.figsize"] = (20, 15)
        plt.show()

        if show_mse:
            base_preds = pd.Series(np.squeeze(self.df.y_scaler.inverse_transform(self.df.base_preds)),
                                   index=self.df.test_idx)
            base_rmse = mean_squared_error(self.sim, base_preds, squared=False)
            self.test_rmse = mean_squared_error(self.sim, self.preds, squared=False)
            print(f"Baseline loss: {base_rmse}")
            print(f"Model loss: {self.test_rmse}")

        return self.preds

    def return_metrics(self):
        """
        Method that calculates extra measures of the test set
        :return: rmse and r^2 measures
        """
        r2 = r2_score(self.preds, self.sim)

        return self.test_rmse, r2

    def plot_predicted_truth_corr(self, labels=['Modelled concentrations', 'LSTM predicted concentrations'], title=None,
                                  color='royalblue'):
        fig, ax = plt.subplots()
        ax.scatter(self.sim, self.preds, s=600, marker='P', color=color, alpha=0.5)
        ax.plot([self.sim.min(), self.sim.max()], [self.sim.min(), self.sim.max()], '--', color='gold', lw=10)
        ax.set_xlabel(labels[0], fontdict={'size': 36})
        ax.set_ylabel(labels[1], fontdict={'size': 36})
        ax.tick_params(axis='both', which='major', labelsize=32)

        if title is not None:
            ax.set_title(title, {'fontsize': 42})

        fig.show()

        return

    def save_model(self, name):
        """
        :param name: name for the saved model
        """
        self.model.save(name)
        print('Model has been saved with a name ', name, '\n')

        return
