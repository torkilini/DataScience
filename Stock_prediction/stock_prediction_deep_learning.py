import os
import secrets
import pandas as pd
from datetime import datetime

from stock_prediction_class import Stock_Prediction
from stock_prediction_lstm import LSTM
from stock_prediction_numpy import Stock_Data
from stock_prediction_plotter import Plot_Functions

def train_LSTM_network(stock):
    data = Stock_Data(stock)
    Plot_Functions = Plot_Functions(True, stock.get_project_folder(), data.get_stock_short_name(), data.get_stock_currency(), stock.get_ticker())
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(TIME_STEPS, stock.get_project_folder())
    Plot_Functions.plot_histogram_data_split(training_data, test_data, stock.get_validation_date())

    lstm = LSTM(stock.get_project_folder())
    model = lstm.create_model(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_defined_metrics())
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
                        callbacks=[lstm.get_callback()])
    print("saving weights")
    model.save(os.path.join(stock.get_project_folder(), 'model_weights.h5'))

    Plot_Functions.plot_loss(history)
    Plot_Functions.plot_mse(history)

    print("display the content of the model")
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    print("plotting prediction results")
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = data.get_min_max().inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
    test_predictions_baseline.to_csv(os.path.join(stock.get_project_folder(), 'predictions.csv'))

    test_predictions_baseline.rename(columns={0: STOCK_TICKER + '_predicted'}, inplace=True)
    test_predictions_baseline = test_predictions_baseline.round(decimals=0)
    test_predictions_baseline.index = test_data.index
    Plot_Functions.project_plot_predictions(test_predictions_baseline, test_data)

    print("prediction is finished")


if __name__ == '__main__':
    STOCK_TICKER = 'ETH-USD'
    STOCK_START_DATE = pd.to_datetime('2015-08-07')
    STOCK_VALIDATION_DATE = pd.to_datetime('2021-09-01')
    EPOCHS = 100
    BATCH_SIZE = 32
    TIME_STEPS = 3
    TODAY_RUN = datetime.today().strftime("%Y%m%d")
    TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
    print('Ticker: ' + STOCK_TICKER)
    print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Test Run Folder: ' + TOKEN)
    # create project run folder
    PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)
    if not os.path.exists(PROJECT_FOLDER):
        os.makedirs(PROJECT_FOLDER)

    stock_prediction = Stock_Prediction(STOCK_TICKER, STOCK_START_DATE, STOCK_VALIDATION_DATE, PROJECT_FOLDER)
    # Execute Deep Learning model
    train_LSTM_network(stock_prediction)
