import os
from absl import app
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from stock_prediction_class import Stock_Prediction
from stock_prediction_numpy import Stock_Data
from datetime import timedelta


def main(argv):
    print(tf.version.VERSION)
    inference_folder = os.path.join(os.getcwd(), RUN_FOLDER)
    stock = Stock_Prediction(STOCK_TICKER, STOCK_START_DATE, STOCK_VALIDATION_DATE, inference_folder)

    data = Stock_Data(stock)

    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(TIME_STEPS, inference_folder)
    min_max = data.get_min_max()

    # load future data

    print('Latest Stock Price')
    latest_close_price = test_data.Close.iloc[-1]
    latest_date = test_data[-1:]['Close'].idxmin()
    print(latest_close_price)
    print('Latest Date')
    print(latest_date)

    tomorrow_date = latest_date + timedelta(1)
    # Specify the next 300 days
    next_year = latest_date + timedelta(TIME_STEPS*100)

    print('Future Date')
    print(tomorrow_date)

    print('Future Timespan Date')
    print(next_year)

    x_test, y_test, test_data = data.generate_future_data(TIME_STEPS, min_max, tomorrow_date, next_year, latest_close_price)

    # load the weights from our best model
    model = tf.keras.models.load_model(os.path.join(inference_folder, 'model_weights.h5'))
    model.summary()

    #print(x_test)
    #print(test_data)
    # display the content of the model
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    # perform a prediction
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = min_max.inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)

    test_predictions_baseline.rename(columns={0: STOCK_TICKER + '_predicted'}, inplace=True)
    test_predictions_baseline = test_predictions_baseline.round(decimals=0)
    test_data.to_csv(os.path.join(inference_folder, 'generated.csv'))
    test_predictions_baseline.to_csv(os.path.join(inference_folder, 'inference.csv'))

    print("plotting predictions")
    plt.figure(figsize=(14, 5))
    plt.plot(test_predictions_baseline[STOCK_TICKER + '_predicted'], color='red', label='Predicted [' + 'GOOG' + '] price')
    plt.xlabel('Time')
    plt.ylabel('Price [' + 'USD' + ']')
    plt.legend()
    plt.title('Prediction')
    plt.savefig(os.path.join(inference_folder, STOCK_TICKER + '_future_prediction.png'))
    plt.pause(0.001)

    plt.figure(figsize=(14, 5))
    plt.plot(test_data.Close, color='green', label='Simulated [' + 'GOOG' + '] price')
    plt.xlabel('Time')
    plt.ylabel('Price [' + 'USD' + ']')
    plt.legend()
    plt.title('Random')
    plt.savefig(os.path.join(inference_folder, STOCK_TICKER + '_future_random.png'))
    plt.pause(0.001)
    plt.show()


if __name__ == '__main__':
    TIME_STEPS = 3
    RUN_FOLDER = 'Data_folder'
    STOCK_TICKER = 'GOOG'
    STOCK_START_DATE = pd.to_datetime('2004-08-01')
    STOCK_VALIDATION_DATE = pd.to_datetime('2017-01-01')
    app.run(main)