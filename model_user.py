from model_creator import Model_Creator
from historical_data import get_all_historical_data
import datetime as dt

model = Model_Creator()

model.epochs = 1
model.batch_size = 1
model.days_anticipated = 60
model.train_percentaje = 0.8

currency = "BTCUSDT"
df = get_all_historical_data(currency,dt.datetime(2013,12,1))

model.load_dataset(df, currency)
model.prepare_dataset()
model.create_model()
predictor = model.train_model()
model.test_model()
