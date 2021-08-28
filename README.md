# Breathe in Breathe out
 Repository created for the "Breathe in Breathe out: Deep Learning and Data Assimilation for a Big Data problem" MSc Individual Project
 
 `Pipeline` class is the final workflow developed in the project, however other classes like `DNA_regressor`, `LSTM_regressor`, `Dataset` and etc can still be used separately without being involved in the `Pipeline` class.
 
 `Pipeline` object requires three files as an input - historical observed and modelled values for the LSTM and DNA training, latest observed values for the final data assimilation and industrial data file for the correlation measures. Currently default parameters are chosen to match our dataset. The full pipeline execution then looks something like this:
 ```Python
 # LSTM and DNA model configurations
 lstm_config = {'seq_length': 13, 'neurons': [40, 40], 'lr': 0.01, 'inp_shape': 5}
 dna_config = {'seq_length': 16, 'neurons': [45, 20], 'lr': 0.01, 'inp_shape': 6}
 
 #initialisation, can also be done with pretrained models instead by passing the models as parameters
 pipeline = Pipeline('hist_data.csv', 'obs_data.csv', 'industries.csv', lstm_config, dna_config)
 
 pipeline.train_pipeline(50, 50) #LSTM and DNA model training
 pipeline.generate_model_predictions() #LSTM and DNA predictions
 pipeline.generate_assimilations() #Optimal Interpolation assimilations
 pipeline.plot_corr() #correlation calculation and plotting
 ```
 
 Example of `LSTM_regressor` training without the `Pipeline`
 ```Python
 #preparing input dataset where df is pandas dataframe of historical modelled values
 dataset = Dataset(df, n_input=22, scaled_cols_oth=None)
 
 regressor = LSTM_regressor(dataset, n_units=[40, 40], inp_shape=5)
 regressor.fit(epochs=20)
 regressor.predict()
 ```
 
 
 Main dependencies are `tensorflow`, `numpy`, `sklearn`, `pandas` and in case hyperparameter optimization is needed - `keras-tuner`.
 
 ![Alt text](/misc/classes.png?raw=true "Class scheme and relationships")
 
##### Sadly data used in the project is confidential, so no actual results can be shown here

