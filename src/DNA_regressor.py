from OptInterpolation import *
from sklearn.metrics import r2_score
from Regressor import *
import tensorflow as tf


class DNA_regressor(Regressor):
    def predict(self, plot=False, plot_length=120):
        self.preds = self.model.predict(self.df.test_gen)
        if plot:
            self.obs = pd.Series(
                self.df.y_scaler.inverse_transform(np.squeeze(self.df.test_data_out['obs'].values[self.df.n_input:])),
                index=self.df.test_idx)
            self.sim = pd.Series(
                self.df.y_scaler.inverse_transform(np.squeeze(self.df.test_data_out['sim'].values[self.df.n_input:])),
                index=self.df.test_idx)
            self.preds = pd.Series(self.df.y_scaler.inverse_transform(np.squeeze(self.preds)), index=self.df.test_idx)

            plt.plot(self.preds[:min(plot_length, len(self.preds))], 'b--', label='DNA predicted')
            plt.plot(self.obs[:min(plot_length, len(self.obs))], 'r-', label='Observations')
            plt.plot(self.sim[:min(plot_length, len(self.sim))], 'y-.', label='AMS-MINNI modelled')
            plt.legend()
            plt.show()
        return self.preds  # , base_mse, test_mse

    def dna_metrics(self):
        sim = self.sim.values
        obs = self.obs.values
        preds = self.preds.values
        mse_f = np.sqrt(np.square(sim - obs).sum()) / np.sqrt(np.square(obs).sum())
        mse_a = np.sqrt(np.square(preds - obs).sum()) / np.sqrt(np.square(obs).sum())
        return mse_f, mse_a


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    location = 'C:\\Users\\Nikodemas\\Desktop\\IMPERIAL material\\MAGIS\\duomenys\\NO2_data_final.dat'
    df1 = pd.read_csv(location)
    df = df1.copy()
    df_brun = df[df['sta_name'] == 'P.S.GIOVANNI']
    df_brun.sort_values(by=['date'])
    # # df_brun.loc[:, 'date'] = pd.to_datetime(df_brun.loc[:, 'date'], format='%Y-%m-%d %H:%M')
    # # df_brun = df_brun.replace(-999.0000, np.nan)
    # # df_brun['day_date'] = pd.to_datetime(df_brun['date']).dt.date
    # # df_brun['obs'] = df_brun['obs'].fillna(df_brun.groupby('day_date')['obs'].transform('median'))
    # # df_brun['sim'] = df_brun['sim'].fillna(df_brun.groupby('day_date')['sim'].transform('median'))
    # # while df_brun['obs'].isna().sum():
    # #     df_brun['obs'] = df_brun['obs'].fillna(df_brun['obs'].shift(24))
    # # while df_brun['sim'].isna().sum():
    # #     df_brun['sim'] = df_brun['sim'].fillna(df_brun['sim'].shift(24))
    # # df_brun = df_brun.set_index('date')
    # scaler = StandardScaler()  # MinMaxScaler(feature_range=(-1, 1))
    # dataset = Dataset(df_brun, 8736, 48, 48, ['sim', 'day_sin', 'day_cos'], ['sim'], 12, 24, scaler, scaler, scaled_cols_oth=None)
    #
    # model = Sequential()
    # model.add(LSTM(50, return_sequences=True, input_shape=(None, 3)))
    # model.add(LSTM(35, return_sequences=False))
    # model.add(Dense(1))
    #
    #
    # regressor = Regressor(dataset, model)
    # regressor.fit(2)
    # preds, base_loss, model_loss = regressor.predict()
    # print(f"Baseline loss: {base_loss}")
    # print(f"Model loss: {model_loss}")

    df_brun = clean_na(df_brun, ['sim', 'obs'], 'median', -999.0000)
    df_brun07 = df_brun[(df_brun.index >= '2007-01-01') & (df_brun.index < '2008-01-01')]
    print(df_brun07.index.max(), df_brun07.index.min())

    df_brun = add_day_trig(df_brun)
    df_brun = add_month_trig(df_brun)

    scaler = StandardScaler()  # MinMaxScaler(feature_range=(-1, 1))
    dataset = Dataset(df_brun, 52608, 8760, 8760,
                      ['sim', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
                      ['sim', 'obs'],
                      19, 24, scaler, scaler)

    # regressor = Regressor(dataset, 40, 45, 5)
    regressor = DNA_regressor(dataset, [40, 45], 0.001, 5)

    regressor.fit(1)
    predictions = regressor.predict()
    print(predictions)

    # oi_scaler = StandardScaler()
    # scaled_pred = oi_scaler.fit_transform(predictions.reshape(-1, 1))
    df_obs = pd.read_csv(
        'C:\\Users\\Nikodemas\\Desktop\\IMPERIAL material\\MAGIS\\duomenys\\giov_no2.csv')
    df_obs = df_obs.sort_values(by=['DatetimeBegin'], ascending=True)[['Concentration', 'DatetimeBegin']]
    df_obs.columns = ['obs', 'date']
    df_obs = clean_na(df_obs, ['obs'], 'mean')
    data_ass = OptInterpolation(df_obs[['obs']].values,
                                predictions.reshape(-1, 1),
                                df_obs.index)

    updates13 = data_ass.assimilate(R=1, plot=True)

    # scaler = StandardScaler()  # MinMaxScaler(feature_range=(-1, 1))
    # dataset = Dataset(df_brun, 52608, 8760, 8760,
    #                   ['sim', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
    #                   ['sim'],
    #                   19, 24, scaler, scaler, scaled_cols_oth=None)
    #
    # regressor = Regressor(dataset, [40, 45], 0.001, 5)

    # regressor.fit(5, plot=True)
    # regressor.predict(show_mse=True)
    df_brun07 = df_brun[(df_brun.index >= '2007-01-01') & (df_brun.index < '2008-01-01')]
    data_ass07 = OptInterpolation(df_brun07[['obs']].values, df_brun07[['sim']].values, df_brun07[['obs']].index)
    updates07 = data_ass07.assimilate(R=1, plot=True)
    import pickle

    df_brun10 = df_brun[(df_brun.index >= '2010-01-01') & (df_brun.index < '2011-01-01')]
    data_ass10 = OptInterpolation(df_brun10[['obs']].values, df_brun10[['sim']].values, df_brun10[['obs']].index)
    updates10 = data_ass10.assimilate(R=1, plot=True)

    d2013 = np.array([x.mean() for x in updates13]).mean()
    d2010 = np.array([x.mean() for x in updates10]).mean()
    d2007 = np.array([x.mean() for x in updates07]).mean()
    x = np.array([d2007, d2010, d2013])
    y = np.array([1156, 1201, 1676])
    print(f"x: {x}")
    print(f"y: {y}")
    plt.plot(x, y, 'o')

    m, b = np.polyfit(x, y, 1)

    plt.plot(x, m * x + b)
    plt.xlabel('average pollution')
    plt.ylabel('number of industries')
    plt.show()

    filename = 'saved_model.sav'

    regressor.model.save('my_model.h5')
    # # preds, base_loss, model_loss = regressor.predict()
    # # print(f"Baseline loss: {base_loss}")
    # # print(f"Model loss: {model_loss}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
