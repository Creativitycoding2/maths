# stock_predictor/views.py
import base64
import io

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg

import pandas as pd
from django.shortcuts import render
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')


def upload_file(request):
    return render(request, 'drop.html')


def stock_prediction(request):
    """csv_file_path = finders.find('stocks.csv')
    data = pd.read_csv(csv_file_path)"""
    data2 = request.FILES['fileUpload']
    data = pd.read_csv(data2)
    dates = data['Date']
    # Remove the "Date" column
    # data.drop(columns=['Date'], inplace=True)
    # data.drop(columns=['Volume'], inplace=True)
    # Remove commas from numerical columns and convert to numeric format
    numeric_columns = ['Open', 'High', 'Low', 'Close']
    for col in numeric_columns:
        try:
            # Split the string by comma, convert each value to float, and then join them back
            data[col] = data[col].astype(str).str.split(',').apply(lambda x: [float(i) for i in x]).apply(
                lambda x: sum(x) / len(x))
        except TypeError:
            # If conversion fails, ignore the column
            data.drop(columns=[col], inplace=True)

    # Display the cleaned dataframe

    # Save the cleaned dataframe to a new CSV file
    data.to_csv('cleaned_stock_data.csv', index=False)
    data['Price_Change'] = data['Close'].diff()  # Difference between consecutive closing prices

    # Define a threshold
    threshold = 0

    # Make binary predictions
    data['Prediction'] = data['Price_Change'].apply(lambda x: 1 if x > threshold else 0)

    # Calculate probabilities
    positive_prob = data['Prediction'].mean()
    negative_prob = 1 - positive_prob
    # Step 1: Data Collection
    # Load the dataset
    data = pd.read_csv('cleaned_stock_data.csv', usecols=numeric_columns)

    # Step 2: Data Preprocessing
    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Split the data into features (X) and target variable (y)
    X = data.drop(columns=['Open'])
    y = data['Open']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Choose a Regression Algorithm
    # Initialize the regression model
    model = RandomForestRegressor()
    # Step 5: Train the Model
    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Step 6: Evaluate the Model
    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE) to evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    # Plot the data
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(dates, data['Open'], label='Open', color='blue')
    ax.plot(dates, data['High'], label='High', color='green')
    ax.plot(dates, data['Low'], label='Low', color='red')
    ax.plot(dates, data['Close'], label='Close', color='orange')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('NFT Stock Prices')
    ax.legend()

    ax.tick_params(axis='x', rotation=-30)

    # Save the plot to a buffer
    buffer = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buffer)
    plt.close(fig)

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    fig1, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.bar(['Positive', 'Negative'], [positive_prob, negative_prob], color=['green', 'red'])
    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability of Price Movement')

    # Save the plot to a buffer
    buffer1 = io.BytesIO()
    canvas1 = FigureCanvasAgg(fig1)
    canvas1.print_png(buffer1)
    plt.close(fig1)

    # Convert the image to a base64 string
    encoded_image1 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    # Convert the image to a base64 string
    params = {'mse': mse, 'enc': encoded_image, 'enc1': encoded_image1, 'positive_prob': str(float(positive_prob)*100)+'%', 'negative_prob': str(float(negative_prob)*100)+'%'}
    return render(request, 'stock_predictor.html', params)
