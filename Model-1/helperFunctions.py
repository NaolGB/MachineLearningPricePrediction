from http import client
from os import fwalk
import pandas as pd
from utils import config
from binance import Client
import talib as ta
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

client = Client(api_key=config.APIKEY, api_secret=config.SECRETKEY)

def fetchBianceData(ticker :str, timeInterval :str, startDate :str, endDate :str, stableCoin='USDT',  generateFor='training'):
    if timeInterval=='15min' or timeInterval=='15Min' or timeInterval=='15m':
        interval = Client.KLINE_INTERVAL_15MINUTE
    elif timeInterval=='1hr' or timeInterval=='1Hr' or timeInterval=='1h':
        interval = Client.KLINE_INTERVAL_1HOUR
    elif timeInterval=='4hr' or timeInterval=='4Hr' or timeInterval=='4h':
        interval = Client.KLINE_INTERVAL_4HOUR
    elif timeInterval=='8hr' or timeInterval=='8Hr' or timeInterval=='8h':
        interval = Client.KLINE_INTERVAL_8HOUR
    elif timeInterval=='12hr' or timeInterval=='12Hr' or timeInterval=='12h':
        interval = Client.KLINE_INTERVAL_12HOUR
    else:
        return('Please insert valid interval: 15min, 1hr, 4hr, 8hr, 12hr')
    
    try:
        data = client.get_historical_klines(
            f'{ticker}{stableCoin}',
            interval=interval,
            start_str=startDate,
            end_str=endDate
        )

        df = pd.DataFrame(
            data=data,
            columns=['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numOfTrades', 'takerBuyBaseAssetVolume', 'takerBuyQuoteAssetVolume', '_']
        )

        # generateFor used to differentiate in saving the data, also mentioned in generateFeatures
        if generateFor.lower() == 'training' or generateFor.lower() == 'testing':
            df.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_{generateFor.lower()}_raw.csv', index=False)

            print(f'Data fetched and succesfully saved at data/{ticker}{stableCoin}_{timeInterval}_{generateFor.lower()}_raw.csv')
            
            return pd.read_csv(f'data/{ticker}{stableCoin}_{timeInterval}_{generateFor.lower()}_raw.csv')
        else:
            print('Please provice propoer generateFor value: "training", "testing"')
            return('Please provice propoer generateFor value: "training", "testing"')

    except Exception as e:
        return(f'Data fetch failed {e}')


def generateFeatures(ticker :str, timeInterval :str, predictionLength :int, historicalLength :int, stableCoin='USDT', generateFor='training'):
    df = pd.read_csv(
        f'data/{ticker}{stableCoin}_{timeInterval}_{generateFor.lower()}_raw.csv',
            usecols=['open', 'high', 'low', 'close']
        )
    ema = ta.EMA(df['close'], timeperiod=9)
    df['ema'] = ema

    for i in range(len(df) - predictionLength):
        df.loc[i, 'trendChange'] = 100*(df.loc[i+predictionLength,'ema'] - df.loc[i, 'ema']) / df.loc[i, 'ema']

    # save the orginal column list to use for building new columns with trasposed data
    initialColumnsList = df.columns.tolist()
    df['percChange'] = round(100*(df['close'] - df['open'])/df['open'], 2)

    newColumnList = [f'{i}_term_percChange'for i in range(1, historicalLength)]
    df = df.reindex(newColumnList + ['percChange'] + initialColumnsList, axis=1)

    # transposing the previous predictionLength values of precChange column for every row
    for i in range(historicalLength, len(df)):
        tempList =  df.loc[i-(historicalLength-1):i-1, 'percChange'].to_numpy().flatten().tolist()
        df.loc[i] = tempList + [df.loc[i, 'percChange']] + list(df.loc[i, initialColumnsList])
    df.dropna(inplace=True)

    # generateFor helps determine whether to keep 'open', 'close', 'high', 'low', 'ema', 'trendChange' some needed for train and others for test
    #   number of the dropped columns also affects in building and using the model, specifically, we slice DataFrames for prediction based on number of these columns
    if generateFor.lower() == 'training':
        df.drop(['open', 'close', 'high', 'low', 'ema'], axis=1, inplace=True)
    elif generateFor.lower() == 'testing':
        df.drop(['ema', 'trendChange'], axis=1, inplace=True)
    else:
        print('Please provice propoer generateFor value: "training", "Testing"')
        return('Please provice propoer generateFor value: "training", "Testing"')

    df.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_features_{generateFor.lower()}.csv', index=False)

    # to reset index for slicing in labeling, send new DF to avoid referencing
    df = pd.read_csv(f'data/{ticker}{stableCoin}_{timeInterval}_features_{generateFor.lower()}.csv')

    return df

def labelDataframeMultilabel(df :pd.DataFrame, ticker :str, timeInterval :str, lowerCut :int, upperCut :int, stableCoin='USDT'):
    # lowerCut and upperCut are determeined, for this implementation, by loofing at the percChange distribution
    for i in range(len(df)):
        if df.loc[i, 'trendChange'] <= lowerCut:
            df.loc[i, 'label'] = 0
        elif df.loc[i, 'trendChange'] >= upperCut:
            df.loc[i, 'label'] = 2
        else:
            df.loc[i, 'label'] = 1

    print('0: downward trend, 1: sideways trend, 2: up trend')
    df.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_multi_label_trended.csv', index=False)

    newDf = pd.read_csv(f'data/{ticker}{stableCoin}_{timeInterval}_multi_label_trended.csv')
    newDf.drop('trendChange', axis=1, inplace=True)

    return newDf

def labelDataframeBinary(df :pd.DataFrame, ticker :str, timeInterval :str, midCut :int, stableCoin='USDT'):
    for i in range(len(df)):
        if df.loc[i, 'trendChange'] <= midCut:
            df.loc[i, 'label'] = 0
        else:
            df.loc[i, 'label'] = 1

    print('0: downward trend, 1: up trend')
    df.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_binary_trended.csv', index=False)
    
    newDf = pd.read_csv(f'data/{ticker}{stableCoin}_{timeInterval}_binary_trended.csv')
    newDf.drop('trendChange', axis=1, inplace=True)

    return newDf

def generateTrainTestDatasetBinary(df :pd.DataFrame, ticker :str, timeInterval :str, stableCoin='USDT', testSize=0.3):
    numberOfClasses=2
    train, test = train_test_split(df, test_size=testSize) # random_seed not set, set if needed
    
    xTrain = train.drop('label', axis=1)
    yTrain = train['label']
    yTrain = np_utils.to_categorical(y=yTrain, num_classes=numberOfClasses) # changed to categorical for categorical_crossentropy loss metric, comment out possible for different metrics
    yTrain = pd.DataFrame(yTrain, dtype=int)

    xTest = test.drop('label', axis=1)
    yTest = test['label']
    yTest = np_utils.to_categorical(y=yTest, num_classes=numberOfClasses)
    yTest = pd.DataFrame(yTest, dtype=int)

    xTrain.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_binary_train_features.csv', index=False)
    yTrain.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_binary_train_label.csv', index=False)

    xTest.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_binary_test_features.csv', index=False)
    yTest.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_binary_test_label.csv', index=False)

    print('Binary xTrain, yTrain, xTest, yTest saved in data and returned in this order')

    return(xTrain, yTrain, xTest, yTest)


def generateTrainTestDatasetMultiLabel(df :pd.DataFrame, ticker :str, timeInterval :str, numberOfClasses = 3, stableCoin='USDT', testSize=0.3):
    train, test = train_test_split(df, test_size=testSize)
    
    xTrain = train.drop('label', axis=1)
    yTrain = train['label']
    yTrain = np_utils.to_categorical(y=yTrain, num_classes=numberOfClasses)
    yTrain = pd.DataFrame(yTrain, dtype=int)

    xTest = test.drop('label', axis=1)
    yTest = test['label']
    yTest = np_utils.to_categorical(y=yTest, num_classes=numberOfClasses)
    yTest = pd.DataFrame(yTest, dtype=int)

    xTrain.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_multi_label_train_features.csv', index=False)
    yTrain.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_multi_label_train_label.csv', index=False)

    xTest.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_multi_label_test_features.csv', index=False)
    yTest.to_csv(f'data/{ticker}{stableCoin}_{timeInterval}_multi_label_test_label.csv', index=False)

    print('Multi label xTrain, yTrain, xTest, yTest saved in data and returned in this order')

    return(xTrain, yTrain, xTest, yTest)