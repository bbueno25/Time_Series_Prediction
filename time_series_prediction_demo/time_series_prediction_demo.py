"""
DOCSTRING
"""
import math
import matplotlib
import numpy
import pandas
import sklearn.metrics
import statsmodels
import warnings

warnings.filterwarnings("ignore")
matplotlib.pyplot.style.use('fivethirtyeight')

class AR_Model:
    """
    DOCSTRING
    """
    def __call__(self):
        matplotlib.pyplot.figure(figsize=(15, 8))
        model = statsmodels.tsa.arima_model.ARIMA(Train_log, order=(2, 1, 0))
        results_AR = model.fit(disp=-1)
        matplotlib.pyplot.plot(train_log_diff.dropna(), label="Original")
        matplotlib.pyplot.plot(results_AR.fittedvalues, color='red', label='Predictions')
        matplotlib.pyplot.legend(loc='best')
        AR_predict = results_AR.predict(start="2014-06-25", end="2014-09-25")
        AR_predict = AR_predict.cumsum().shift().fillna(0)
        AR_predict1 = pandas.Series(numpy.ones(
            valid.shape[0]) * numpy.log(valid['Count'])[0], index=valid.index)
        AR_predict1=AR_predict1.add(AR_predict, fill_value=0)
        AR_predict = numpy.exp(AR_predict1)
        matplotlib.pyplot.figure(figsize=(15, 8))
        matplotlib.pyplot.plot(valid['Count'], label="Validation")
        matplotlib.pyplot.plot(AR_predict, color="red", label="Predict")
        matplotlib.pyplot.legend(loc="best")
        matplotlib.pyplot.title('RMSE: %.4f' % (
            numpy.sqrt(numpy.dot(AR_predict, valid['Count']))/valid.shape[0]))
        matplotlib.pyplot.show()
        matplotlib.pyplot.figure(figsize=(15, 8))
        model = statsmodels.tsa.arima_model.ARIMA(Train_log, order=(0, 1, 2))
        results_MA = model.fit(disp=-1)
        matplotlib.pyplot.plot(train_log_diff.dropna(), label="Original")
        matplotlib.pyplot.plot(results_MA.fittedvalues, color="red", label="Prediction")
        matplotlib.pyplot.legend(loc="best")
        MA_predict = results_MA.predict(start="2014-06-25", end="2014-09-25")
        MA_predict=MA_predict.cumsum().shift().fillna(0)
        MA_predict1=pandas.Series(numpy.ones(
            valid.shape[0])*numpy.log(valid['Count'])[0], index=valid.index)
        MA_predict1=MA_predict1.add(MA_predict,fill_value=0)
        MA_predict = numpy.exp(MA_predict1)
        matplotlib.pyplot.figure(figsize=(15, 8))
        matplotlib.pyplot.plot(valid['Count'], label="Valid")
        matplotlib.pyplot.plot(MA_predict, color='red', label="Predict")
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.title('RMSE: %.4f' % (
            numpy.sqrt(numpy.dot(MA_predict, valid['Count']))/valid.shape[0]))
        matplotlib.pyplot.show()
 
class ARIMA_Model():
    """
    DOCSTRING
    """
    def __call__(self):
        matplotlib.pylab.rcParams['figure.figsize']=(20, 10)
        test_stationary(train_original['Count'])

    def differencing(self):
        """
        DOCSTRING
        """
        train_log_diff = Train_log - Train_log.shift(1)
        test_stationary(train_log_diff.dropna())

    def remove_seasonality(self):
        """
        DOCSTRING
        """
        matplotlib.pyplot.figure(figsize=(16, 10))
        decomposition = statsmodels.tsa.seasonal.seasonal_decompose(
            pandas.DataFrame(Train_log).Count.values, freq=24)
        matplotlib.pyplot.style.use('default')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        matplotlib.pyplot.subplot(411)
        matplotlib.pyplot.plot(Train_log, label='Original')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.subplot(412)
        matplotlib.pyplot.plot(trend, label='Trend')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.subplot(413)
        matplotlib.pyplot.plot(seasonal, label='Seasonal')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.subplot(414)
        matplotlib.pyplot.plot(residual, label='Residuals')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.tight_layout()

    def remove_trend(self):
        """
        DOCSTRING
        """
        Train_log = numpy.log(Train['Count'])
        valid_log = numpy.log(valid['Count'])
        moving_avg = pandas.rolling_mean(Train_log, 24)
        matplotlib.pyplot.plot(Train_log)
        matplotlib.pyplot.plot(moving_avg, color='red')
        train_log_moving_diff = Train_log - moving_avg
        train_log_moving_diff.dropna(inplace=True)
        test_stationary(train_log_moving_diff)

    def stationarity_of_residuals(self):
        """
        DOCSTRING
        """
        matplotlib.pyplot.figure(figsize=(16, 8))
        train_log_decompose = pandas.DataFrame(residual)
        train_log_decompose['date'] = Train_log.index
        train_log_decompose.set_index('date', inplace=True)
        train_log_decompose.dropna(inplace=True)
        test_stationary(train_log_decompose[0])

    def test_stationary(self, timeseries):
        """
        DOCSTRING
        """
        # determine rolling statistics
        rolmean = pandas.rolling_mean(timeseries, window=24)
        rolstd = pandas.rolling_std(timeseries, window=24)
        # plot rolling Statistics
        orig = matplotlib.pyplot.plot(timeseries, color="blue", label="Original")
        mean = matplotlib.pyplot.plot(rolmean, color="red", label="Rolling Mean")
        std = matplotlib.pyplot.plot(rolstd, color="black", label="Rolling Std")
        matplotlib.pyplot.legend(loc="best")
        matplotlib.pyplot.title("Rolling Mean and Standard Deviation")
        matplotlib.pyplot.show(block=False)
        # perform Dickey Fuller test
        print("Results of Dickey Fuller test: ")
        dftest = statsmodels.tsa.stattools.adfuller(timeseries, autolag='AIC')
        dfoutput = pandas.Series(
            dftest[0:4], index=[
                'Test Statistics', 'p-value', '# Lag Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' %key] = value
        print(dfoutput)

    def autocorrelation(self):
        """
        DOCSTRING
        """
        lag_acf = statsmodels.tsa.stattools.acf(train_log_diff.dropna(), nlags=25)
        lag_pacf = statsmodels.tsa.stattools.pacf(
            train_log_diff.dropna(), nlags=25, method="ols")
        matplotlib.pyplot.figure(figsize=(15, 8))
        matplotlib.pyplot.style.use("fivethirtyeight")
        matplotlib.pyplot.plot(lag_acf)
        matplotlib.pyplot.axhline(y=0, linestyle="--", color="gray")
        matplotlib.pyplot.axhline(
            y=-1.96/numpy.sqrt(len(train_log_diff.dropna())), linestyle="--", color="gray")
        matplotlib.pyplot.axhline(
            y=1.96 /numpy.sqrt(len(train_log_diff.dropna())), linestyle="--", color="gray")
        matplotlib.pyplot.title("Autocorrelation Function")
        matplotlib.pyplot.show()
        # PACF
        matplotlib.pyplot.figure(figsize=(15, 8))
        matplotlib.pyplot.plot(lag_pacf)
        matplotlib.pyplot.axhline(y=0, linestyle="--", color="gray")
        matplotlib.pyplot.axhline(
            y=-1.96/numpy.sqrt(len(train_log_diff.dropna())), linestyle="--", color="gray")
        matplotlib.pyplot.axhline(
            y=1.96/numpy.sqrt(len(train_log_diff.dropna())), linestyle="--", color="gray")
        matplotlib.pyplot.title("Partial Autocorrelation Function")
        matplotlib.pyplot.show()

class CombinedModel:
    """
    DOCSTRING
    """
    def __call__(self):
        matplotlib.pyplot.figure(figsize=(16, 8))
        model = statsmodels.tsa.arima_model.ARIMA(Train_log, order=(2, 1, 2))
        results_ARIMA = model.fit(disp=-1)
        matplotlib.pyplot.plot(train_log_diff.dropna(), label='Original')
        matplotlib.pyplot.plot(results_ARIMA.fittedvalues, color='red', label='Predicted')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.show()
        ARIMA_predict_diff = results_ARIMA.predict(start="2014-06-25", end="2014-09-25")
        matplotlib.pyplot.figure(figsize=(16, 8))
        check_prediction_diff(ARIMA_predict_diff, valid)

    def check_prediction_diff(self, predict_diff, given_set):
        """
        DOCSTRING
        """
        predict_diff= predict_diff.cumsum().shift().fillna(0)
        predict_base = pandas.Series(numpy.ones(
            given_set.shape[0]) * numpy.log(given_set['Count'])[0], index=given_set.index)
        predict_log = predict_base.add(predict_diff, fill_value=0)
        predict = numpy.exp(predict_log)
        matplotlib.pyplot.plot(given_set['Count'], label="Given set")
        matplotlib.pyplot.plot(predict, color='red', label="Predict")
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.title('RMSE: %.4f' % (
            numpy.sqrt(numpy.dot(predict, given_set['Count']))/given_set.shape[0]))
        matplotlib.pyplot.show()

    def check_prediction_log(self, predict_log, given_set):
        """
        DOCSTRING
        """
        predict = numpy.exp(predict_log)
        matplotlib.pyplot.plot(given_set['Count'], label="Given set")
        matplotlib.pyplot.plot(predict, color='red', label="Predict")
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.title('RMSE: %.4f' % (
            numpy.sqrt(numpy.dot(predict, given_set['Count']))/given_set.shape[0]))
        matplotlib.pyplot.show()

class DataProcessing:
    """
    DOCSTRING
    """
    def __init__(self):
        train = pandas.read_csv("Train_SU63ISt.csv")
        test = pandas.read_csv("Test_0qrQsBZ.csv")

    def applyer(self, row):
        """
        DOCSTRING
        """
        if row.dayofweek == 5 or row.dayofweek == 6:
            return 1
        return 0

    def create_copy(self):
        """
        DOCSTRING
        """
        train_original = train.copy()
        test_original = test.copy()
        train.columns, test.columns
        train.dtypes, test.dtypes
        train.shape, test.shape

    def divide_data(self):
        """
        DOCSTRING
        """
        Train = train.ix['2012-08-25':'2014-06-24']
        valid = train.ix['2014-06-25':'2014-09-25']
        Train.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14, label='Train')
        valid.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14, label='Valid')
        matplotlib.pyplot.xlabel('Datetime')
        matplotlib.pyplot.ylabel('Passenger Count')
        matplotlib.pyplot.legend(loc='best')
    
    def exploratory_analysis(self):
        """
        DOCSTRING
        """
        train.groupby('year')['Count'].mean().plot.bar()
        train.groupby('month')['Count'].mean().plot.bar()
        temp = train.groupby(['year', 'month'])['Count'].mean()
        temp.plot(figsize =(15,5), title = "Passenger Count(Monthwise)", fontsize = 14)
        train.groupby('day') ['Count'].mean().plot.bar()
        train.groupby('Hour')['Count'].mean().plot.bar()
        train.groupby('weekend') ['Count'].mean().plot.bar()
        train.groupby('Day of week') ['Count'].mean().plot.bar()
        train.Timestamp = pandas.to_datetime(train.Datetime, format = '%d-%m-%y %H:%M')
        train.index = train.Timestamp
        hourly = train.resample('H').mean()
        daily = train.resample('D').mean()
        weekly = train.resample('W').mean()
        monthly = train.resample('M').mean()
        fig, axs = matplotlib.pyplot.subplots(4,1)
        hourly.Count.plot(figsize = (15, 8), title = "Hourly", fontsize = 14, ax = axs[0])
        daily.Count.plot(figsize = (15, 8), title = "Daily", fontsize = 14, ax = axs[1])
        weekly.Count.plot(figsize = (15, 8), title = "Weekly", fontsize = 14, ax = axs[2])
        monthly.Count.plot(figsize = (15, 8), title = "Monthly", fontsize = 14, ax = axs[3])
        test.Timestamp = pandas.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
        test.index = test.Timestamp
        # converting to daily mean 
        test = test.resample('D').mean()
        train.Timestamp = pandas.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
        train.index = train.Timestamp
        # converting to daily mean
        train = train.resample('D').mean()

    def extract_features(self):
        """
        DOCSTRING
        """
        train['Datetime'] = pandas.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
        test['Datetime'] = pandas.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
        train_original['Datetime'] = pandas.to_datetime(
            train_original.Datetime, format='%d-%m-%Y %H:%M')
        test_original['Datetime'] = pandas.to_datetime(
            test_original.Datetime, format='%d-%m-%Y %H:%M')
        for i in (train, test, train_original, test_original):
            i['year'] = i.Datetime.dt.year
            i['month'] = i.Datetime.dt.month
            i['day'] = i.Datetime.dt.day
            i['Hour'] = i.Datetime.dt.hour
        train['Day of week'] = train['Datetime'].dt.dayofweek
        temp = train['Datetime']
        temp2 = train['Datetime'].apply(applyer)
        train['weekend'] = temp2
        train.index = train['Datetime']
        df = train.drop('ID', 1)
        ts = df['Count']
        matplotlib.pyplot.figure(figsize=(16, 8))
        matplotlib.pyplot.plot(ts)
        matplotlib.pyplot.title("Time Series")
        matplotlib.pyplot.xlabel("Time (year-month)")
        matplotlib.pyplot.ylabel("Passenger Count")
        matplotlib.pyplot.legend(loc='best')
 
class HoltLinearModel:
    """
    DOCSTRING
    """
    def __call__(self):
        matplotlib.pyplot.style.use('default')
        matplotlib.pyplot.figure(figsize=(16, 8))
        statsmodels.api.tsa.seasonal_decompose(Train.Count).plot()
        result = statsmodels.api.tsa.stattools.adfuller(train.Count)
        matplotlib.pyplot.show()
        y_hat_holt = valid.copy()
        fit1 = statsmodels.tsa.api.Holt(numpy.asarray(Train['Count'])).fit(
            smoothing_level=0.3, smoothing_slope=0.1)
        y_hat_holt['Holt_linear'] = fit1.forecast(len(valid))
        matplotlib.pyplot.style.use('fivethirtyeight')
        matplotlib.pyplot.figure(figsize=(15, 8))
        matplotlib.pyplot.plot(Train.Count, label='Train')
        matplotlib.pyplot.plot(valid.Count, label='Validation')
        matplotlib.pyplot.plot(y_hat_holt['Holt_linear'], label='Holt Linear')
        matplotlib.pyplot.legend(loc='best')
        rmse = math.sqrt(sklearn.metrics.mean_squared_error(
            valid.Count, y_hat_holt.Holt_linear))
        print(rmse)

    def predictions(self):
        """
        DOCSTRING
        """
        predict = fit1.forecast(len(test))
        test['prediction'] = predict
        # calculating hourly ration of count
        train_original['ratio'] = train_original['Count']/train_original['Count'].sum()
        # grouping hourly ratio
        temp = train_original.groupby(['Hour'])['ratio'].sum()
        # group by to csv format
        pandas.DataFrame(temp, columns=['Hour', 'ratio']).to_csv('Groupby.csv')
        temp2 = pandas.read_csv("Groupby.csv")
        temp2 = temp2.drop('Hour.1', 1)
        # merge test and test_original on day, month and year
        merge = pandas.merge(
            test, test_original, on=('day', 'month', 'year'), how='left')
        merge['Hour'] = merge['Hour_y']
        merge = merge.drop(
            ['year', 'month', 'day', 'Hour_x', 'Datetime', 'Hour_y'], axis=1)
        # predicting by merging temp2 and merge
        prediction = pandas.merge(merge, temp2, on='Hour', how='left')
        # converting the ration to original scale
        prediction['Count'] = prediction['prediction'] * prediction['ratio'] * 24
        prediction['ID'] = prediction['ID_y']
        prediction.head()
        submission = prediction.drop(['ID_x', 'ID_y', 'prediction', 'Hour', 'ratio'], axis=1)
        pandas.DataFrame(submission, columns = ['ID','Count']).to_csv('Holt_Linear.csv')
        
class HoltWinterModel:
    """
    DOCSTRING
    """
    def __call__(self):
        y_hat_avg = valid.copy()
        fit1 = statsmodels.tsa.api.ExponentialSmoothing(numpy.asarray(
            Train['Count']), seasonal_periods=7, trend='add', seasonal='add').fit()
        y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid))
        matplotlib.pyplot.figure(figsize=(16, 8))
        matplotlib.pyplot.plot(Train['Count'], label='Train')
        matplotlib.pyplot.plot(valid['Count'], label='Validation')
        matplotlib.pyplot.plot(y_hat_avg.Holt_Winter, label='Holt Winters')
        matplotlib.pyplot.legend(loc='best')
        rmse = math.sqrt(
            sklearn.metrics.mean_squared_error(valid['Count'], y_hat_avg['Holt_Winter']))
        print(rmse)

    def predictions(self):
        """
        DOCSTRING
        """
        predict = fit1.forecast(len(test))
        test['prediction'] = predict
        # merge test and test_original on day, month, and year
        merge = pandas.merge(test, test_original, on=('day', 'month', 'year'), how='left')
        merge['Hour'] = merge['Hour_y']
        merge.head()
        merge = merge.drop(['year', 'month', 'Datetime', 'Hour_x', 'Hour_y'], axis=1)
        # predicting by merge and temp2
        prediction = pandas.merge(merge, temp2 , on='Hour', how='left')
        # converting the ration to original scale
        prediction['Count'] = prediction['prediction'] * prediction['ratio'] * 24
        prediction.head()
        prediction['ID'] = prediction['ID_y']
        submission = prediction.drop(
            ['ID_x', 'ID_y', 'day', 'Hour', 'prediction', 'ratio'], axis=1)
        pandas.DataFrame(submission, columns = ['ID', 'Count']).to_csv('Holt winters.csv')

class MovingAverageModel:
    """
    DOCSTRING
    """
    def __call__(self):
        y_hat_avg = valid.copy()
        y_hat_avg['moving_average_forecast'] = Train['Count'].rolling(10).mean().iloc[-1]
        matplotlib.pyplot.figure(figsize=(15, 5))
        matplotlib.pyplot.plot(Train['Count'], label='Train')
        matplotlib.pyplot.plot(valid['Count'], label='Validation')
        matplotlib.pyplot.plot(
            y_hat_avg['moving_average_forecast'],
            label='Moving Average Forecast with 10 Observations')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.show()
        y_hat_avg = valid.copy()
        y_hat_avg['moving_average_forecast'] = Train['Count'].rolling(20).mean().iloc[-1]
        matplotlib.pyplot.figure(figsize=(15, 5))
        matplotlib.pyplot.plot(Train['Count'], label='Train')
        matplotlib.pyplot.plot(valid['Count'], label='Validation')
        matplotlib.pyplot.plot(
            y_hat_avg['moving_average_forecast'],
            label='Moving Average Forecast with 20 Observations')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.show()
        y_hat_avg = valid.copy()
        y_hat_avg['moving_average_forecast'] = Train['Count'].rolling(50).mean().iloc[-1]
        matplotlib.pyplot.figure(figsize = (15, 5))
        matplotlib.pyplot.plot(Train['Count'], label='Train')
        matplotlib.pyplot.plot(valid['Count'], label='Validation')
        matplotlib.pyplot.plot(
            y_hat_avg['moving_average_forecast'],
            label="Moving Average Forecast with 50 Observations")
        matplotlib.pyplot.legend(loc = 'best')
        matplotlib.pyplot.show()
        rmse = math.sqrt(sklearn.metrics.mean_squared_error(
            valid['Count'], y_hat_avg['moving_average_forecast']))
        print(rmse)

class NaiveModel:
    """
    DOCSTRING
    """
    def __call__(self):
        dd = numpy.asarray(Train.Count)
        y_hat = valid.copy()
        y_hat['naive'] = dd[len(dd)-1]
        matplotlib.pyplot.figure(figsize=(12, 8))
        matplotlib.pyplot.plot(Train.index, Train['Count'], label='Train')
        matplotlib.pyplot.plot(valid.index, valid['Count'], label='Validation')
        matplotlib.pyplot.plot(y_hat.index, y_hat['naive'], label='Naive')
        matplotlib.pyplot.legend(loc='best')
        matplotlib.pyplot.title('Naive Forecast')
        rmse = math.sqrt(sklearn.metrics.mean_squared_error(valid.Count, y_hat.naive))
        print(rmse)

class SARIMAX_Model:
    """
    DOCSTRING
    """
    def __call__(self):
        y_hat_avg = valid.copy()
        fit1 = statsmodels.api.tsa.statespace.SARIMAX(
            Train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
        y_hat_avg['SARIMA'] = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True)
        matplotlib.pyplot.figure(figsize=(16, 8))
        matplotlib.pyplot.plot(Train['Count'], label="Train")
        matplotlib.pyplot.plot(valid.Count, label="Validation")
        matplotlib.pyplot.plot(y_hat_avg['SARIMA'], label="SARIMA")
        matplotlib.pyplot.legend(loc="best")
        matplotlib.pyplot.title("SARIMAX Model")
        rms = sqrt(sklearn.metrics.mean_squared_error(valid.Count, y_hat_avg.SARIMA))
        print(rms)

    def hourly_predictions(self):
        """
        DOCSTRING
        """
        predict = fit1.predict(start="2014-9-26", end="2015-4-26", dynamic=True)
        test['prediction'] = predict
        # merge test and test_original on day, month, and year
        merge = pandas.merge(test, test_original, on=('day', 'month', 'year'), how='left')
        merge['Hour'] = merge['Hour_y']
        # predicting by merging merge and temp2
        prediction = pandas.merge(merge, temp2, on='Hour', how='left')
        # converting the ratio to original scale
        prediction['Count'] = prediction['prediction'] * prediction['ratio'] * 24
        prediction['ID'] = prediction['ID_y']
        submission=prediction.drop(['day', 'Hour', 'ratio', 'prediction', 'ID_x', 'ID_y'], axis=1)
        # converting the final submission to csv format
        pandas.DataFrame(submission, columns=['ID','Count']).to_csv('SARIMAX.csv')

class SimpleExponentialModel:
    """
    DOCSTRING
    """
    def __call__(self):
        y_hat = valid.copy()
        fit2 = statsmodels.tsa.api.SimpleExpSmoothing(numpy.asarray(
            Train['Count'])).fit(smoothing_level=0.6, optimized=False)
        y_hat['SES'] = fit2.forecast(len(valid))
        matplotlib.pyplot.figure(figsize=(15, 8))
        matplotlib.pyplot.plot(Train['Count'], label='Train')
        matplotlib.pyplot.plot(valid['Count'], label='Validation')
        matplotlib.pyplot.plot(y_hat['SES'], label='Simple Exponential Smoothing')
        matplotlib.pyplot.legend(loc='best')
        rmse = math.sqrt(sklearn.metrics.mean_squared_error(valid.Count, y_hat['SES']))
        print(rmse)

if __name__ == '__main__':
    data_processing = DataProcessing()
    data_processing.create_copy()
    data_processing.extract_features()
    data_processing.exploratory_analysis()
    data_processing.divide_data()
    NaiveModel()
    SimpleExponentialModel()
    holt_linear_model = HoltLinearModel()
    holt_linear_model.predictions()
    holt_winter_model = HoltWinterModel()
    holt_winter_model.predictions()
    arima_model = ARIMA_Model()
    arima_model.remove_trend()
    arima_model.differencing()
    arima_model.remove_seasonality()
    arima_model.stationarity_of_residuals()
    arima_model.autocorrelation()
    AR_Model()
    combined_model = CombinedModel()
    combined_model.check_prediction_diff()
    combined_model.check_prediction_log()
    sarimax_model = SARIMAX_Model()
    sarimax_model.hourly_predictions()
