# forex-forecasting 
This package is for the development of models to forecast forex. The package 
itself is within the early development stage. It currently works using a single
main script that loads data from various csv files and tests the forecast 
performance of various machine learning and econometric models using backtesting. 


### Step-by-step setup and run main scripts from command line 
1) Download forex data and regressor data from 
https://drive.google.com/drive/folders/189yWLvS_tV4PaPWEpEOO57F6eb9peWbM?usp=share_link
2) Configure config files to point to downloaded data or put data in package 
directory and leave config file as is
3) Ensure package directory is working directory
4) run setup.py develop 
5run main.py config/config.ini (or own config file) to see comparison of 
multiple models forecasting performance



