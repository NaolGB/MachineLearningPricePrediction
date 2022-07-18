# MachineLearningPricePrediction
Price prediction for cryptocurrency using machine learning, reinforcement learning in plan

### TODO
- [x] Model 1 (BTCUSDT, historical terms 256, time intrval 1hr, prediction length 12hr)
    - [x] Exploratory Data Analysys
    - [x] Model Building
      - [x] Binary CNN1D
      - [x] Multi class CNN1D 
    - [x] Training
    - [x] Back Testing


## About
Price prediction by classification. I performed binary classification and multilabel classificaiton on cryptocurrency market data. I have used Convolutional Neural Network for classification in this public repository. I am doing further research and trying different types of modes including Logistic Regression, LSTM, KNN, SVM, and Ensemble methods. 

The back testing reveals that the models and strategy currently used here are not profitable. So, I am working on creating better strategies, models and working on reinforcement learning to produce highly profitable system.  

## Possible Points of Upgrade
This you can find with in the codes as a comment. These are points of upgrade I imagined could better the program.This are suggestions, but also my focus as I improve this program.

## Visualizations
Visualizations for this project are made in Tableau because of the high quality visualizations Tableau creates. However, DataFrames are easily accessible from multiple points in the program so it is easy to plot any plot with Matplotlib.

The Tableau files along with the data are found in data/Model-1 Visualizations.twb or see the interactive versuion in Table Public <a href="https://public.tableau.com/shared/8CT4DNX6R?:display_count=n&:origin=viz_share_link">here for Binary Classification </a> and <a href="https://public.tableau.com/views/CryptocurrencyPricePredictionModel-1Visualizations-MultiLabel/MultiLabel?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link">here for Multi label Classification</a>.

The non-interactive versions are given below for fast access
![Binary percentage change](Model-1/data/Binary_perc_change.png)
![Binary labels](Model-1/data/Binary_labels.png)
![Binary fitting](Model-1/data/Binary_fitting.png)
![Binary back testing](Model-1/data/Binary_back_testing.png)

![MultiLabel percentage change](Model-1/data/MultiLabel_prec_change.png)
![MultiLabel labels](Model-1/data/MultiLabel_labels.png)
![MultiLabel fitting](Model-1/data/MultiLabel_fitting.png)
![MultiLabel back testing](Model-1/data/MultiLabel_back_testing.png)

## Current/Future Progress
Currently, under this project, I am working on:
- sentiment analysis
- multi-model strategy
- high frequency trading strategy