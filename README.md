# forex-forecasting 
This package is for the development of models to forecast forex.


### Step-by-step setup and run main scripts from command line 
1) Download forex data from Yahoo: https://finance.yahoo.com/quote/COP=X
2) Configure config files <br />
&nbsp; a) config/config.ini <br />
&nbsp; &nbsp; i) "general-source_type" - dir for directory, csv for csv file <br />
&nbsp; &nbsp; ii) "general-path" - paths to forex data directory or csv file
3) Ensure package directory is working directory
4) run main.py config/config.ini (or own config file)

### Package development requirements

Macro-economic implementation: 
1. acquire taylor rule fundamental data
2. implement basic taylor rule fundamental model (Rossi)
3. random walk comparison

LSTM implementation
1. implement LSTM model
2. random walk comparison

Sentiment analysis implementation
1. trading economics news parser
2. raw data to vector preparation
3. BERT model implementation
4. sentiment over time plotting

Model selection and adaptive Hedge
1. implement adaptive Hedge model



