import pandas as pd

def preprocess_and_split(df, features, target_col='total_amount', seed=42):
    # drop missing
    df = df.dropna(subset=features + [target_col]).copy()

    # datetime features
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'],
                                     format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'],
                                      format="%Y-%m-%d %H:%M:%S", errors='coerce')

    df['trip_duration_minutes'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60.0
    df = df[df['trip_duration_minutes'] <= 300]  # filter long trips
    df['pickup_hour'] = df['pickup_dt'].dt.hour

    # bucket pickup hour
    def demand_bucket(hour):
        if pd.isna(hour): return "normal"
        if 2 <= hour < 6: return "low"
        elif 6 <= hour < 16: return "normal"
        else: return "high"
    df['pickup_hour_bucket'] = df['pickup_hour'].apply(demand_bucket)
    pickup_dummies = pd.get_dummies(df['pickup_hour_bucket'], prefix="pickup_hour")
    df = pd.concat([df, pickup_dummies], axis=1)

    # filter target (trim 5–95%)
    lower = df[target_col].quantile(0.05)
    upper = df[target_col].quantile(0.95)
    df = df[(df[target_col] >= lower) & (df[target_col] <= upper)].copy()

    # drop unused
    df = df.drop(columns=['pickup_dt','dropoff_dt','pickup_hour','pickup_hour_bucket'])

    # split train/test
    N = len(df)
    rng = pd.np.random.RandomState(seed)  # or numpy RandomState
    perm = rng.permutation(N)
    n_train = int(0.7 * N)
    train = df.iloc[perm[:n_train]].reset_index(drop=True)
    test = df.iloc[perm[n_train:]].reset_index(drop=True)
    return train, test

def normalize_train_test(train_df, test_df):
    target = 'total_amount' if 'total_amount' in train_df.columns else 'total amount'
    X_cols = [c for c in train_df.columns if c != target]
    stats = {}
    for c in X_cols:
        mean = train_df[c].mean()
        std = train_df[c].std() or 1.0
        train_df[c] = (train_df[c] - mean) / std
        test_df[c] = (test_df[c] - mean) / std
        stats[c] = (mean, std)
    # normalize target
    y_mean = train_df[target].mean()
    y_std = train_df[target].std() or 1.0
    train_df[target] = (train_df[target] - y_mean) / y_std
    test_df[target] = (test_df[target] - y_mean) / y_std
    stats['__target__'] = (y_mean, y_std)
    return train_df, test_df, stats, X_cols, target
