from historical_data import get_all_historical_data
import datetime as dt

if __name__ == "__main__":
    currency = "BTCUSDT"
    btc_hist = get_all_historical_data(currency,dt.datetime(2020,12,1))
    print(btc_hist)