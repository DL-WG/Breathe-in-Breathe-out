from Regressor import *
from DNA_regressor import *
from Dataset import *
from OptInterpolation import *
from PearsonCorr import *
from utils import *

plt.rcParams["figure.figsize"] = (20, 15)

"""
Implementation of the full workflow developed in the project.
It takes in three data files - one with historical
observed and modelled values, one with the latest observations that need to be assimilated
and one with industrial data (e. g. number of industries in region) and outputs
correlation between air and industrial data.
"""


class Pipeline:
    def __init__(self, full_inp, obs_inp, corr_inp, LSTM_param, DNA_param, LSTMmodel=None, DNAmodel=None,
                 input_cols=['sim', 'day_sin', 'day_cos', 'month_sin', 'month_cos'], obs_label='obs', sim_label='sim',
                 train_size=52608, val_size=8760, test_size=8760):
        """
        Object can be initialised with new configuration parameters for the ML models or with already pretrained
        models
        :param full_inp: historical obs+sim data
        :param obs_inp: latest observed data
        :param corr_inp: industrial data
        :param LSTM_param: LSTM model configuration
        :param DNA_param: DNA model configuration
        :param LSTMmodel: already trained LSTM model
        :param DNAmodel: already trained DNA model
        """
        self.input_cols = input_cols
        self.sim_label = sim_label
        self.obs_label = obs_label

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        df = pd.read_csv(full_inp)
        df = clean_na(df, [sim_label, obs_label], 'mean', -999.0000)
        self.df07 = df[(df.index >= '2007-01-01') & (df.index < '2008-01-01')]
        self.df10 = df[(df.index >= '2010-01-01') & (df.index < '2011-01-01')]
        df = add_day_trig(df)
        self.df = add_month_trig(df)

        df_obs = pd.read_csv(obs_inp)
        df_obs = df_obs.sort_values(by=['DatetimeBegin'], ascending=True)[['Concentration', 'DatetimeBegin']]
        df_obs.columns = [obs_label, 'date']
        self.df_obs = clean_na(df_obs, [obs_label], 'mean')

        self.y_corr = pd.read_csv(corr_inp).columns.astype(int)

        self.LSTM = self.create_LSTM(LSTM_param, LSTMmodel)
        self.DNA = self.create_DNA(DNA_param, DNAmodel)

    def create_LSTM(self, par, model):
        """
        Method that creates LSTM model
        :param par: LSTM configuration
        :param model: pretrained model
        :return: LSTM regressor object
        """
        scaler = StandardScaler()
        dataset = Dataset(self.df, self.train_size, self.val_size, self.test_size,
                          self.input_cols,
                          [self.sim_label],
                          par['seq_length'], 24, scaler, scaler, scaled_cols_oth=None, base_pred=True)
        # return Regressor(dataset, par['neurons'], par['lr'], par['inp_shape'])
        return Regressor(dataset, par['neurons'], par['lr'], par['inp_shape'], model)

    def create_DNA(self, par, model):
        """
        Method that creates DNA model
        :param par: DNA configuration
        :param model: pretrained model
        :return: DNA regressor object
        """
        scaler = StandardScaler()
        dataset = Dataset(self.df, self.train_size, self.val_size, self.test_size,
                          self.input_cols,
                          [self.sim_label, self.obs_label],
                          par['seq_length'], 24, scaler, scaler)
        return DNA_regressor(dataset, par['neurons'], par['lr'], par['inp_shape'], model)

    def train_pipeline(self, LSTM_epochs, DNA_epochs, LSTM_plot=False, DNA_plot=False):
        """
        Method responsible for the model training
        :param LSTM_epochs: num of LSTM training epochs
        :param DNA_epochs: num of DNA training epochs
        :param LSTM_plot: plot LSTM training curve
        :param DNA_plot: plot DNA training curve
        """
        if LSTM_epochs > 0:
            print('Training LSTM model\n')
            self.LSTM.fit(LSTM_epochs, plot=LSTM_plot)

        if DNA_epochs > 0:
            print('Training DNA model\n')
            self.DNA.fit(DNA_epochs, plot=DNA_plot)

        return

    def generate_model_predictions(self, LSTM_plot=False, DNA_plot=True):
        """
        Method responsible for the predictions
        :param LSTM_plot: plot prediction vs ground truth graph
        :param DNA_plot: plot simulations vs ground truth vs assimilations graph
        """
        self.LSTM_preds = self.LSTM.predict(show_mse=LSTM_plot)
        self.DNA_preds = self.DNA.predict(plot=DNA_plot)
        return

    def generate_assimilations(self, R07=5000, R10=5000, R13=5000, plot07=False, plot10=False, plot13=False):
        """

        :param R07: covariance of the first DA
        :param R10: covariance of the first DA
        :param R13: covariance of the first DA
        :param plot07: plot simulations vs ground truth vs assimilations graph
        :param plot10: plot simulations vs ground truth vs assimilations graph
        :param plot13: plot simulations vs ground truth vs assimilations graph
        """
        self.data_ass07 = OptInterpolation(self.df07[['obs']].values, self.df07[['sim']].values,
                                           self.df07[['obs']].index)
        self.updates07 = self.data_ass07.assimilate(R=R07, plot=plot07)
        self.data_ass10 = OptInterpolation(self.df10[['obs']].values, self.LSTM_preds.values.reshape(-1, 1),
                                           self.df10[['obs']].index)
        self.updates10 = self.data_ass10.assimilate(R=R10, plot=plot10)
        self.data_ass13 = OptInterpolation(self.df_obs[['obs']].values, self.DNA_preds.values.reshape(-1, 1),
                                           self.df_obs[['obs']].index)
        self.updates13 = self.data_ass13.assimilate(R=R13, plot=plot13,
                                                    labels=['Optimal interpolation', 'Observations', 'DNA predicted'])

    def plot_corr(self, y=None, labels=['Average pollution', 'Number of industries'], title=None):
        """
        Plots and calculates correlation between industrial and air data
        :param y: list of industrial values if different from init part is needed
        :param labels: axis labels in the graph
        :param title: graph title
        :return: correlation coefficient between air quality vs industrial data
        """
        d2013 = np.array([x.mean() for x in self.updates13]).mean()
        d2010 = np.array([x.mean() for x in self.updates10]).mean()
        d2007 = np.array([x.mean() for x in self.updates07]).mean()

        self.avg_pollution = [d2007, d2010, d2013]

        if y is None:
            y = self.y_corr

        self.corr = PearsonCorr(self.avg_pollution, y)
        self.corr.plot_corr(labels, title)

        return self.corr.correlation()

# {'seq_length':5, 'neurons':[40, 30], 'lr':0.01, 'inp_shape':5}
