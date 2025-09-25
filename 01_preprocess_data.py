import pandas as pd
from datetime import datetime
import argparse

def log_time(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
def log_data_frame_length(df):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - current Length: {len(df)}")

# -------- Entry point --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script with arguments.")
    parser.add_argument("--input_path", type=str, help="Input dataset path", default='sample_data/nytaxi2022.csv')
    parser.add_argument("--output_path", type=str, help="Output cleaned data path", default='sample_data/cleaned_data_v4.csv')

    args = parser.parse_args()
    log_time(f"argument data {args}")
    # Load dataset
    requested_features = [
            'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
            'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type', 'extra'
        ]
    target_col='total_amount'
    focus_columns = requested_features + [target_col]
    # -------- Step 1: Load Data set --------
    log_time(" 1. Starting to load data")
    df = pd.read_csv(args.input_path, usecols=focus_columns , parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], date_format='%m/%d/%Y %I:%M:%S %p')
    log_data_frame_length(df)
    log_time(" 1. Finished loading data")
    # -------- Step 2: drop na columns --------
    df.dropna(inplace=True)
    log_data_frame_length(df)
    log_time(" 2. Finished drop na data")

    # -------- Step 3: Filter trips with invalid extra --------
    df = df[(df['extra'] >= 0)]
    log_data_frame_length(df)
    log_time(" 3. Filtering trips with valid extra")

    # -------- Step 4: Filter trip_distance --------
    # log_time("Filtering trip_distance")
    # remove near-zero distances , filter to show only > 0.1
    # remove unrealistic long trips, filter to show only <100
    df = df[(df["trip_distance"] > 0.1) & (df["trip_distance"] < 100)]
    log_data_frame_length(df)
    log_time(" 4. Filtering trips distance")

    # -------- Step 5: Calculate trip_duration_minutes and filter  --------
    # log_time("Calculating trip duration in minutes")
    df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Filter trips with duration between 1 and 300 minutes
    df = df[(df['trip_duration_minutes'] >= 2) & (df['trip_duration_minutes'] <= 300)]
    log_data_frame_length(df)
    log_time(" 5. Filtering trips with valid duration")

    # -------- Step 6: Filter extreme fares (keep 3%–97%)--------
    #lower = df[target_col].quantile(0.03)
    #upper = df[target_col].quantile(0.97)
    #df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    #log_data_frame_length(df)
    #log_time(" 6. Final Filter extreme fares")

    # -------- Step 7: Bucket demand group  --------
    # Extract pickup hour
    # log_time("Extracting pickup hour and assigning demand group")
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

    def demand_group(hour):
        if 14 <= hour <= 18:
            return "high"
        elif 8 <= hour <= 13 or 19 <= hour <= 22:
            return "normal"
        else:
            return "low"

    df['demand_type'] = df['pickup_hour'].apply(demand_group)
    pickup_dummies = pd.get_dummies(df['demand_type'], prefix="demand_type")
    df = pd.concat([df, pickup_dummies], axis=1)
    log_time(" 7. group demand by pickup hour")

    # -------- Step 8: Bucket passenger_count --------
    # log_time("Bucketing passenger_count")
    df['passenger_count_single'] = (df['passenger_count'] == 1).astype(int)
    df['passenger_count_double'] = (df['passenger_count'] == 2).astype(int)
    df['passenger_count_other'] = ((df['passenger_count'] != 1) & (df['passenger_count'] != 2)).astype(int)
    log_time(" 8. group passenger count")

    # -------- Step 9: Bucket RatecodeID --------
    # log_time("Bucketing RatecodeID")
    df['RatecodeID_1'] = (df['RatecodeID'] == 1).astype(int)
    df['RatecodeID_other'] = (df['passenger_count'] != 1).astype(int)
    log_time(" 9. group RatecodeID")

    # -------- Step 10: Bucket payment_type --------
    # log_time("Bucketing payment_type")
    df['payment_type_1'] = (df['payment_type'] == 1).astype(int)
    df['payment_type_2'] = (df['payment_type'] == 2).astype(int)
    df['payment_type_other'] = ((df['payment_type'] != 1) & (df['payment_type'] != 2)).astype(int)
    log_time("10. group payment_type")

    # -------- Step 11: Create PULocationID_rank --------
    # log_time("Computing PULocationID rank")
    pul_freq = df['PULocationID'].value_counts()
    quantiles = pul_freq.quantile([i/10 for i in range(1, 10)]).values
    def assign_rank(location_id):
        freq = pul_freq.get(location_id, 0)
        rank = 1
        for q in quantiles:
            if freq > q:
                rank += 1
            else:
                break
        return rank
    df['PULocationID_rank'] = df['PULocationID'].apply(assign_rank)
    log_time("11. Computing PULocationID rank")

    # -------- Step 12 (DOLocationID): Repeat for DOLocationID --------
    # log_time("Computing DOLocationID rank")
    dol_freq = df['DOLocationID'].value_counts()
    quantiles_dol = dol_freq.quantile([i/10 for i in range(1, 10)]).values

    def assign_dol_rank(location_id):
        freq = dol_freq.get(location_id, 0)
        rank = 1
        for q in quantiles_dol:
            if freq > q:
                rank += 1
            else:
                break
        return rank

    df['DOLocationID_rank'] = df['DOLocationID'].apply(assign_dol_rank)
    log_time("12. Computing DOLocationID rank")

    # -------- Step 13: Drop temporary columns --------
    # log_time("Dropping temporary columns")
    cols_to_drop = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_hour', 'demand_type',
        'passenger_count', 'RatecodeID', 'payment_type',
        'PULocationID', 'DOLocationID'
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    log_time("13. Drop temporary columns")


    # Save cleaned data to new CSV
    df.to_csv(args.output_path, index=False)
    log_time("14. Save cleaned data")
