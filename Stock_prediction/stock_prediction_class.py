

class Stock_Prediction:
    def __init__(self, ticker, start_date, validation_date, project_folder):
        self._ticker = ticker
        self._start_date = start_date
        self._validation_date = validation_date
        self._project_folder = project_folder

    def get_ticker(self):
        return self._ticker

    def set_ticker(self, value):
        self._ticker = value

    def get_start_date(self):
        return self._start_date

    def set_start_date(self, value):
        self._start_date = value

    def get_validation_date(self):
        return self._validation_date

    def set_validation_date(self, value):
        self._validation_date = value

    def get_project_folder(self):
        return self._project_folder

    def set_project_folder(self, value):
        self._project_folder = value