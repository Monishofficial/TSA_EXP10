# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### Date: 01-11-2025

### AIM:
To implement SARIMA model using python.

### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
   
### PROGRAM:
##### Developed By: MONISH N
##### Reg No: 212223240097

Import necessary library:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

```
Load and clean data:
```
data = pd.read_csv('/content/FINAL_USO.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_ts = data['USO_Close']
```
Plot GDP Trend:
```
plt.figure(figsize=(12, 6))
plt.plot(data_ts)
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('Gold Price Time Series (Close)')
plt.grid(True)
plt.show()
```
Check Stationarity :
```
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

print("\n--- Stationarity Check on Closing Prices ---")
check_stationarity(data_ts)

```

Plot ACF and PCF:
```
plt.figure(figsize=(12, 5))
plot_acf(data_ts, lags=50)
plt.title('Autocorrelation Function (ACF) - Google Close Price')
plt.show()
plt.figure(figsize=(12, 5))
plot_pacf(data_ts, lags=50)
plt.title('Partial Autocorrelation Function (PACF) - Google Close Price')
plt.show()

```
Split data:


```
train_size = int(len(data_ts) * 0.8)
train, test = data_ts[:train_size], data_ts[train_size:]

```
Fit SARIMA model:
```
print("\n--- Fitting SARIMA(1, 1, 1)x(1, 1, 1, 5) model ---")
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)

```
Make predictions& Evaluate RMSE:

```
predictions = sarima_result.predict(start=len(train), end=len(data_ts) - 1) 

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
```
Plot Predictions:
```
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title(f'SARIMA Model Predictions (RMSE: {rmse:.3f})')
plt.legend()
plt.grid(True)
plt.show()

```
### OUTPUT:
Original Data:
<img width="1016" height="702" alt="image" src="https://github.com/user-attachments/assets/76df8c86-45d8-49ff-9f87-163888fb1b1a" />

Autocorrelation:
<img width="593" height="458" alt="image" src="https://github.com/user-attachments/assets/25c4dbf4-8119-421b-bb48-fab2334007a6" />

Partial Autocorrelation:
<img width="596" height="460" alt="image" src="https://github.com/user-attachments/assets/6fd6fc86-ccb2-413d-b885-ae9aa66e2d22" />

SARIMA Model:
<img width="1005" height="566" alt="image" src="https://github.com/user-attachments/assets/1bb258d0-50f4-41e2-9764-acb465c406a1" />

RMSE Value:

<img width="586" height="214" alt="image" src="https://github.com/user-attachments/assets/8196568f-e97b-47bf-86c5-77264c44d7ef" />


### RESULT:
Thus the program run successfully based on the SARIMA model.



