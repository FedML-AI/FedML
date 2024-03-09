# evaluate an ARIMA model using a walk-forward validation
import matplotlib.pyplot as plt
import math
import os

from pandas import read_csv
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# load dataset
def parser(x):
    return datetime.strptime(x, "%Y-%m-%dT%H:%M:%Sz")


distribution = "random"
data_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/qps_test_{}_distribution.csv".format(distribution))
plot_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "plot/sample_arima_{}_p{}_d{}_q{}.png")

series = read_csv(data_file,
                  header=0,
                  index_col=0,
                  parse_dates=True,
                  date_parser=parser)
series.index = series.index.to_period('min')
# split into train and test sets
X = series.values
train_size = int(len(X) * 0.1)
train, test = X[:train_size], X[train_size:]
history = [x for x in train]
y = test

# make first prediction
predictions = list()
# p: The lag order, representing the number of lag observations incorporated in the model.
#    In other words, how many past observations we need to consider.
# d: Degree of differencing, denoting the number of times raw observations undergo differencing.
#    We need this to make the time series stationary (similar mean/variance over time).
# q: Order of moving average, indicating the size of the moving average window.
p, d, q = 15, 1, 0
model = ARIMA(history[-100:], order=(p, d, q))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])

# rolling forecasts
for i in range(1, len(y)):
    # predict
    model = ARIMA(history[-100:], order=(p, d, q))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)

# report performance
mse = mean_squared_error(y, predictions)
print('MSE: ' + str(mse))
mae = mean_absolute_error(y, predictions)
print('MAE: ' + str(mae))
rmse = math.sqrt(mean_squared_error(y, predictions))
print('RMSE: ' + str(rmse))

plt.figure(figsize=(16, 8))
# plt.plot_date(series.index[-test_size:], series['qps'].tail(test_size), color='green', label = 'Train')
# plt.plot_date(series.index[test_size:], y, color = 'red', label = 'Real')
# plt.plot_date(series.index[test_size:], predictions, color = 'blue', label = 'Predicted')

plt.scatter([i for i in range(len(series.values))][:train_size], series.values[:train_size], color='green',
            label='Train')
plt.scatter([train_size + i for i in range(len(y))], y, color='red', label='Real')
plt.scatter([train_size + i for i in range(len(predictions))], predictions, color='blue', label='Predicted')
plt.title('QPS Forecasting')
plt.xlabel('Time')
plt.ylabel('QPS')
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.savefig(plot_file.format(distribution, p, d, q))
