from Regressor import *
from DNA_regressor import *
from Dataset import *
from OptInterpolation import *
from PearsonCorr import *
from utils import *

plt.rcParams["figure.figsize"] = (20, 15)


class Pipeline:
    def __init__(self, full_inp, obs_inp, LSTM_param, DNA_param, LSTMmodel=None, DNAmodel=None):
        df = pd.read_csv(full_inp)
        df = clean_na(df, ['sim', 'obs'], 'mean', -999.0000)
        self.df07 = df[(df.index >= '2007-01-01') & (df.index < '2008-01-01')]
        self.df10 = df[(df.index >= '2010-01-01') & (df.index < '2011-01-01')]
        df = add_day_trig(df)
        self.df = add_month_trig(df)
        df_obs = pd.read_csv(obs_inp)
        df_obs = df_obs.sort_values(by=['DatetimeBegin'], ascending=True)[['Concentration', 'DatetimeBegin']]
        df_obs.columns = ['obs', 'date']
        self.df_obs = clean_na(df_obs, ['obs'], 'mean')
        self.LSTM = self.create_LSTM(LSTM_param, LSTMmodel)
        self.DNA = self.create_DNA(DNA_param, DNAmodel)
        # if LSTMmodel is None:
        #     self.LSTM = self.create_LSTM(LSTM_param)
        # else:
        #     self.LSTM = LSTMmodel
        # if DNAmodel is None:
        #     self.DNA = self.create_DNA(DNA_param)
        # else:
        #     self.DNA = DNAmodel

    def create_LSTM(self, par, model):
        scaler = StandardScaler()
        dataset = Dataset(self.df, 52608, 8760, 8760,
                          ['sim', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
                          ['sim'],
                          par['seq_length'], 24, scaler, scaler, scaled_cols_oth=None, base_pred=True)
        # return Regressor(dataset, par['neurons'], par['lr'], par['inp_shape'])
        return Regressor(dataset, par['neurons'], par['lr'], par['inp_shape'], model)

    def create_DNA(self, par, model):
        scaler = StandardScaler()
        dataset = Dataset(self.df, 52608, 8760, 8760,
                          ['sim', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
                          ['sim', 'obs'],
                          par['seq_length'], 24, scaler, scaler)
        return DNA_regressor(dataset, par['neurons'], par['lr'], par['inp_shape'], model)

    def train_pipeline(self, LSTM_epochs, DNA_epochs, LSTM_plot=False, DNA_plot=False):
        if LSTM_epochs > 0:
            print('Training LSTM model\n')
            self.LSTM.fit(LSTM_epochs, plot=LSTM_plot)
        if DNA_epochs > 0:
            print('Training DNA model\n')
            self.DNA.fit(DNA_epochs, plot=DNA_plot)
        return

    def generate_model_predictions(self, LSTM_plot=False, DNA_plot=True):
        self.LSTM_preds = self.LSTM.predict(show_mse=LSTM_plot)
        self.DNA_preds = self.DNA.predict(plot=DNA_plot)
        return

    def generate_assimilations(self, R07=5000, R10=5000, R13=5000, plot07=False, plot10=False, plot13=False):
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

    def plot_corr(self, y, labels, title):
        d2013 = np.array([x.mean() for x in self.updates13]).mean()
        d2010 = np.array([x.mean() for x in self.updates10]).mean()
        d2007 = np.array([x.mean() for x in self.updates07]).mean()
        self.avg_pollution = [d2007, d2010, d2013]
        self.corr = PearsonCorr(self.avg_pollution, y)
        self.corr.plot_corr(labels, title)
        return self.corr.correlation()

# {'seq_length':5, 'neurons':[40, 30], 'lr':0.01, 'inp_shape':5}
