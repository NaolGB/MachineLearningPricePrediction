import pandas as pd
import helperFunctions
import numpy as np

def generateBackTestingData(
    ticker :str, 
    timeInterval :str, 
    startDate :str, 
    endDate :str, 
    predictionLength :int, 
    historicalLength :int, 
    stableCoin='USDT'
    ):

    rawDf = helperFunctions.fetchBianceData(ticker=ticker, timeInterval=timeInterval, startDate=startDate, endDate=endDate, stableCoin=stableCoin, generateFor='testing')
    featuresDf = helperFunctions.generateFeatures(ticker=ticker, stableCoin=stableCoin, historicalLength=historicalLength, predictionLength=predictionLength, timeInterval=timeInterval, generateFor='testing')

    return featuresDf

def predictBianryAndMultilabelOnDataFrame(featureDf, binaryPredictionModel, multiLabelPredictionModel):
    testPredictionsBinary = binaryPredictionModel.predict(featureDf.iloc[:, :-4])
    tredPredictionMultiLabel = multiLabelPredictionModel.predict(featureDf.iloc[:, :-4])

    for i in range(len(featureDf)):
        featureDf.loc[i, 'binaryPrediction'] = ['down', 'up'][np.argmax(testPredictionsBinary[i])]

    for i in range(len(featureDf)):
        featureDf.loc[i, 'multiLabelPrediction'] = ['down', 'side', 'up'][np.argmax(tredPredictionMultiLabel[i])]

    featureDf = featureDf[['open', 'close', 'binaryPrediction', 'multiLabelPrediction']]
    
    print('predictions generated')

    return featureDf


class BackTesting:
    def __init__(self, klineDf :pd.DataFrame, predictionLength :int, tradeFee=0.001, loanFee=0.000004, loanedShares=0, currentStableBalance=1000, currentTickerBalance=0):
        self.tradeFee = tradeFee
        self.loanFee = loanFee
        self.loanedShares = loanedShares
        self.currentStableBalance = currentStableBalance
        self.currentTickerBalance = currentTickerBalance
        self.currentPrice = 0
        self.position = 'none'
        self.klineDf = klineDf
        self.predictionLength = predictionLength
        self.progressDf = pd.DataFrame(columns=[
            'previousStableCoinBalance', 
            'previousTickerBalance', 
            'currentAction', 
            'priceAtCurrentAction', 
            'currentStableCoinBalance', 
            'currentTickerBalance'
            ])


    def goLong(self):
        # TODO
        # (Possible point of upgrade)
        # ---------------------------
        # - add leverage
        self.currentStableBalance -= (self.currentStableBalance * self.tradeFee)

        self.currentTickerBalance = self.currentStableBalance / self.currentPrice
        self.currentStableBalance = 0
        self.position = 'long'

    def goShort(self):
        # TODO
        # (Possible point of upgrade)
        # ---------------------------
        # - add leverage
        self.currentStableBalance -= self.currentStableBalance  * self.tradeFee

        self.loanedShares = self.currentStableBalance / self.currentPrice
        self.currentStableBalance *= 2
        self.position = 'short'


    def closeLongPosition(self):
        # using last index because it ensures all the values can go into one row
        lastIndex = len(self.progressDf)

        # record values before changing
        previousStableCoinBalance = self.currentStableBalance
        previousTickerBalance = self.currentTickerBalance
        currentAction = 'closeLongPosition'
        priceAtCurrentAction = self.currentPrice

        self.currentStableBalance += self.currentTickerBalance * self.currentPrice
        self.currentStableBalance -= self.currentStableBalance * self.tradeFee
        self.currentTickerBalance = 0
        self.position = 'none'

        # record values after changing
        currentStableCoinBalance = self.currentStableBalance
        currentTickerBalance = self.currentTickerBalance

        self.progressDf.loc[lastIndex] = [previousStableCoinBalance, previousTickerBalance, currentAction, priceAtCurrentAction, currentStableCoinBalance, currentTickerBalance]

    def closeShortPosition(self, hours):
        lastIndex = len(self.progressDf)

        # record values before changing
        previousStableCoinBalance = self.currentStableBalance
        previousTickerBalance = self.currentTickerBalance
        currentAction = 'closeShortPosition'
        priceAtCurrentAction = self.currentPrice
        self.currentStableBalance -= (self.loanedShares * self.currentPrice) + (self.loanedShares * self.currentPrice * self.loanFee * hours) + (self.currentStableBalance * self.tradeFee)
        self.position = 'none'

        # record values after changing
        currentStableCoinBalance = self.currentStableBalance
        currentTickerBalance = self.currentTickerBalance

        self.progressDf.loc[lastIndex] = [previousStableCoinBalance, previousTickerBalance, currentAction, priceAtCurrentAction, currentStableCoinBalance, currentTickerBalance]

    def runBackTest(self, predictionType):
        i = 0
        # TODO
        # Possible point of upgrade
        # -------------------------
        # - get out of positions based on OHLC values instead of end of term
        # - multi model decesion (Binary + Multi Label -> Logistic Regression -> long/short)
        if predictionType.lower() == 'binary':
            while i < len(self.klineDf) - self.predictionLength:
                self.currentPrice = self.klineDf.loc[i, 'close']

                if self.position == 'short':
                    self.closeShortPosition(hours=self.predictionLength)

                elif self.position == 'long':
                    self.closeLongPosition()

                if self.position == 'none':
                    if self.currentStableBalance > 0:
                        prediction = self.klineDf.loc[i, 'binaryPrediction']
                        if prediction == 'up':
                            self.goLong()
                        else:
                            self.goShort()
                
                i += self.predictionLength

        elif predictionType.lower() == 'multilabel':
            while i < len(self.klineDf) - self.predictionLength:
                self.currentPrice = self.klineDf.loc[i, 'close']

                if self.position == 'short':
                    self.closeShortPosition(hours=self.predictionLength)

                elif self.position == 'long':
                    self.closeLongPosition()

                if self.position == 'none':
                    if self.currentStableBalance > 0:
                        prediction = self.klineDf.loc[i, 'multiLabelPrediction']
                        if prediction == 'up':
                            self.goLong()
                        elif prediction == 'down':
                            self.goShort()
                
                i += self.predictionLength

        # Finish with ticker balance
        self.currentPrice = self.klineDf.loc[i, 'close']

        if self.position == 'short':
            self.closeShortPosition(hours=self.predictionLength)
        elif self.position == 'long':
            self.closeLongPosition()
        
        # reset_index used to generate dimension index for Tableau
        self.progressDf = self.progressDf.reset_index()
        self.progressDf.to_csv(f'data/progress_{predictionType.lower()}.csv')