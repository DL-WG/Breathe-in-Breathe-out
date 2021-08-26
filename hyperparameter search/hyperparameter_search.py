import kerastuner as kt
from Dataset import *
from datetime import datetime, timedelta
from kerastuner.tuners import RandomSearch
import time
from sklearn.preprocessing import StandardScaler
from tensorflow import *
import os


def build_model(hp):
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(hp.Int('input_units', min_value=10, max_value=60, step=5), return_sequences=True,
                                input_shape=(None, 5)))
    layers_num = hp.Int('n_layers', 0, 4)
    for i in range(layers_num):  # adding variation of layers.
        model.add(keras.layers.LSTM(hp.Int(f'hidden_LSTM_{int(i)}_units', min_value=10, max_value=60, step=5),
                                    return_sequences=True))
    model.add(keras.layers.LSTM(hp.Int('last_LSTM_units', min_value=10, max_value=60, step=5), return_sequences=False))
    model.add(keras.layers.Dense(1))
    hp_learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01,
                                step=0.00005)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate))
    return model


def search_params(df, i, callback, folder='search/', verbose=0):
    LOG_DIR = f"{int(time.time())}"
    n_input = int(i)
    scaler = StandardScaler()
    dataset = Dataset(df, 52608, 8760, 8760,
                      ['sim', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
                      ['sim'],
                      n_input, 24, scaler, scaler, scaled_cols_oth=None)

    tuner = RandomSearch(build_model, objective='val_loss', max_trials=20, executions_per_trial=1,
                         directory=folder + f"length{n_input}_" + LOG_DIR, project_name=f"test_with_length_{i}")

    tuner.search(dataset.train_gen, verbose=verbose, epochs=10, validation_data=dataset.val_gen,
                 callbacks=[callback])
    print(tuner.results_summary(2), "\n\n")


def print_info(strn, start_time, end_time, spent_s, spent_h):
    print(strn)
    print("!!!START TIME: ", start_time)
    print("!!!END TIME: ", end_time)
    print(f"!!!!!!!!!!!!!!!!!!TIME SPENT!!!!!!!!!!!!!!!: {spent_s}\n")
    print(f"!!!!!!!!!!!!!!!!!!TIME SPENT IN HOURS!!!!!!!!!!!!!!!: {spent_h}\n")


if __name__ == '__main__':
    start = time.time()
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    index = int(os.environ['PBS_ARRAY_INDEX'])  # taking process id

    file_num = index % 5 + 1  # assigning specific input file to this process (1, 2, 3, 4 or 5 in this case)
    file_name = 'data/inp' + str(file_num) + '.csv'
    df_h = pd.read_csv(file_name,
                       encoding='ISO-8859-1')
    sta_name = df_h['sta_name'].values[0]
    region = df_h['region'].values[0]
    df_h.sort_values(by=['date'])
    df_h = clean_na(df_h, ['sim'], 'mean', -999.0000)
    df_h = add_day_trig(df_h)
    df_h = add_month_trig(df_h)
    leng = 3 * int((index + 4) / 5) - 2  # assigning specific input size to this process ({1, 4,...,25} in this case)
    strn = "EXPERIMENT WITH SEQUENCE LENGTH " + str(
        leng) + " on file " + file_name + " of " + sta_name + " in region " + region + ":\n"

    search_params(df, leng, stop_early, "no2_search" + str(file_num) + "/", verbose=0)

    end = time.time()
    end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    elapsed = end - start
    print_info(strn, start_time, end_time, elapsed, timedelta(seconds=elapsed))
