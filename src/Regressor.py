from OptInterpolation import *
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.regularizers import l2


class Regressor():

    def __init__(self, df, n_units, lr, inp_shape, model=None):
        self.df = df
        if model is None:
            self.model = self.create_model(n_units, lr, inp_shape)
        else:
            self.model = model
        # self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        # self.model.summary()

    def create_model(self, n_units=[], lr=0.001, inp_shape=5):
        model = Sequential()
        model.add(LSTM(n_units[0], return_sequences=True, input_shape=(None, inp_shape)))#, kernel_regularizer=l2(0.0001)))
        for i in range(len(n_units[1:-1])):
            model.add(LSTM(n_units[i], return_sequences=True))
        model.add(LSTM(n_units[-1], return_sequences=False))
        model.add(Dense(1))
        # model.add(keras.layers.Dense(50, activation='relu'))
        # model.add(keras.layers.Dense(1))
        model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
        model.summary()
        return model

    def fit(self, epochs=20, verbose=True, plot=False, patience=5):
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

    def predict(self, plot_length=120, show_mse=False):
        preds = self.model.predict(self.df.test_gen)
        self.sim = pd.Series(
            self.df.y_scaler.inverse_transform(np.squeeze(self.df.test_data_out['sim'].values[self.df.n_input:])),
            index=self.df.test_idx)
        self.preds = pd.Series(self.df.y_scaler.inverse_transform(np.squeeze(preds)), index=self.df.test_idx)
        plt.plot(self.preds[:min(plot_length, len(self.preds))], 'b--', label='LSTM predicted')
        plt.plot(self.sim[:min(plot_length, len(self.sim))], 'y-.', label='AMS-MINNI modelled')
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
        return self.preds  # , base_mse, test_mse

    def return_metrics(self):
        r2 = r2_score(self.preds, self.sim)
        return self.test_rmse, r2

    def plot_predicted_truth_corr(self, labels=['AMS-MINNI modelled concentrations (' + r'$\mu$g/m$^3$)',
                                                'LSTM predicted concentrations (' + r'$\mu$g/m$^3$)'], title=None,
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

    def save_model(self, name):
        self.model.save(name)
        print('Model has been saved with a name ', name, '\n')
        return


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
    regressor = NA_regressor(dataset, [40, 45], 0.001, 5)

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
