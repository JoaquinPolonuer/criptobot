from model_creator_api import Model_Creator
from historical_data import get_all_historical_data
import datetime as dt

model = Model_Creator()

model.epochs = 4
model.batch_size = 2
model.days_anticipated = 60
model.train_percentaje = 0.9

currency = "BTCUSDT"
df = get_all_historical_data(currency,dt.datetime(2013,12,1))

model.load_dataset(df, currency)
model.prepare_dataset()
model.create_model()
predictor = model.train_model("daily")
model.test_model()
