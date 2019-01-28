import sys
import pandas as pd

def process_dssat_data(input, output):
    df = pd.read_csv(input+"/nbg_maiz_forecast_daily.csv")
    print(df)

if __name__ == "__main__":
    process_dssat_data(sys.argv[1], sys.argv[2])
