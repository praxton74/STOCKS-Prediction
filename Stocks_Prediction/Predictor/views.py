from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf

def predict_price(request):
    return render(request, 'predictor/predict_price.html')

def predict_price_api(request):
    if request.method == 'POST':
        # Get the ticker symbol and end date from the form data
        start_date = '2024-01-01'
        ticker_symbol = request.POST.get('ticker')
        end_date = request.POST.get('end-date')
        print(type(ticker_symbol))
        print(type(end_date))
        # Fetch historical data using yfinance
        MSFT_quote = yf.download(ticker_symbol, start='2024-01-01', end=end_date)

        # Create a new dataframe
        new_df = MSFT_quote.filter(['Adj Close'])

        # Check for missing values and drop them
        new_df = new_df.dropna()

        # Check if you have at least 60 days of data
        if len(new_df) < 60:
            return JsonResponse({'error': 'Insufficient data. Ensure you have at least 60 days of historical data.'})

        # Get the last 60 day closing price values and convert the dataframe to an array
        last_60_days = new_df[-60:].values

        # Scale the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        last_60_days_scaled = scaler.fit_transform(last_60_days)

        # Create an empty list
        X_test = []

        # Append the past 60 days
        X_test.append(last_60_days_scaled)

        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)

        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Get the model path
        if(ticker_symbol=='MSFT'):
            model_path = 'Predictor/one.h5'
        elif(ticker_symbol=='TSLA'):
            model_path = 'Predictor/two.h5'
        elif(ticker_symbol=="GOOG"):
            model_path = 'Predictor/three.h5'
        

        # Load the model
        loaded_model = tf.keras.models.load_model(model_path)

        # Get the predicted scaled price
        pred_price = loaded_model.predict(X_test)

        # Undo the scaling
        pred_price = scaler.inverse_transform(pred_price)

        # Prepare response
        response_data = {'predicted_price': float(pred_price[0][0])}

        return JsonResponse(response_data)

    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'})
