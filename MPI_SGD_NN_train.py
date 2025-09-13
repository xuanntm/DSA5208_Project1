"""
README & Usage (top of file)

This single-file implementation provides an MPI-parallel Stochastic Gradient Descent
trainer for a one-hidden-layer neural network written in Python using mpi4py.

Features:
- One-hidden-layer neural network (configurable hidden neurons)
- Supports activation functions: relu, tanh, sigmoid
- Mini-batch SGD with distributed gradient computation and MPI_Allreduce aggregation
- Local data partitioning (each process reads the CSV and keeps its slice)
- Preprocessing: drops NA, simple datetime features (pickup/dropoff -> duration, pickup hour), label-encoding for categorical fields, normalization
- Train/test split (global, consistent across processes) 70/30
- Reports training and test RMSE, training history (loss vs iteration), and timing

Requirements:
- Python 3.8+
- mpi4py
- numpy
- pandas

Install dependencies:
    pip install mpi4py numpy pandas

Run (example using 4 processes):
    mpiexec -n 4 python MPI_SGD_NN_train.py --data nytaxi2022.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.01 --activation relu

Important notes about parallelism:
- Each process reads the full CSV, but keeps only a roughly-even partition of rows by index: rows where (index % world_size) == rank. This avoids sending datasets between processes during training.
- During training each process forms local mini-batches from its local partition. For each local mini-batch the process computes a local gradient (averaged over its mini-batch). The processes then use MPI_Allreduce to compute the global average gradient across processes. All processes then apply the same weight update so parameters remain synchronized.

Outputs:
- Training history printed to stdout on rank 0
- Final RMSEs printed to stdout on rank 0
- Optionally, you can save model parameters to a npz file by passing --save-model model.npz

"""

import argparse
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
# from collections import defaultdict
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import logging
from contextlib import contextmanager
import psutil

# import warnings
# warnings.filterwarnings("ignore", message="Could not infer format")

# ------------------ Init MPI ------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------ Config Log ------------------
# Create a logger
# Create logger per rank
logger = logging.getLogger(f"rank{rank}")
logger.setLevel(logging.INFO)

# File handler (separate log file per rank)
fh = logging.FileHandler(f"rank{rank}.log", mode="w")
fh.setLevel(logging.INFO)

# Formatter with rank info
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)

# Attach handler (avoid duplicates)
if not logger.handlers:
    logger.addHandler(fh)

# Optional: rank 0 also logs to console
if rank == 0:
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

@contextmanager
def timed_step(logger, rank, step_name):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    logger.info(f"[Rank {rank}] {step_name} took {t1 - t0:.4f} sec")

# ------------------ utilities ------------------

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    g = np.zeros_like(x)
    g[x > 0] = 1.0
    return g

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x)**2

ACTIVATIONS = {
    'relu': (relu, relu_grad),
    'sigmoid': (sigmoid, sigmoid_grad),
    'tanh': (tanh, tanh_grad)
}


def mse_loss_and_grad(y_pred, y_true):
    # returns loss (sum of squared errors / 2) and gradient w.r.t predictions (y_pred)
    err = y_pred - y_true
    loss = 0.5 * np.sum(err**2)
    grad = err
    return loss, grad

# ------------------ neural network ------------------
class OneHiddenNN:
    def __init__(self, input_dim, hidden_dim, activation='relu', seed=1234):
        rng = np.random.RandomState(seed)
        # Xavier init for hidden weights
        # self.W1 = rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / max(1, input_dim))
        self.W1 = rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / max(1, input_dim)) * 0.1
        self.b1 = np.zeros((hidden_dim,))
        # output layer weights
        # self.W2 = rng.randn(hidden_dim) * np.sqrt(2.0 / max(1, hidden_dim))
        self.W2 = rng.randn(hidden_dim) * np.sqrt(2.0 / max(1, hidden_dim)) * 0.1
        self.b2 = 0.0
        self.act_name = activation
        self.act, self.act_grad = ACTIVATIONS[activation]

    def forward(self, X):
        # X: (B, D)
        Z = X.dot(self.W1.T) + self.b1  # (B, H)
        A = self.act(Z)                 # (B, H)
        out = A.dot(self.W2) + self.b2  # (B,)
        cache = (X, Z, A)
        return out, cache

    def backward(self, cache, grad_out):
        X, Z, A = cache
        # grad_out: (B,) gradient of loss w.r.t network output
        B = X.shape[0]
        # gradients w.r.t W2 and b2
        gW2 = (grad_out[:, None] * A).sum(axis=0) / B  # (H,)
        gb2 = grad_out.sum() / B
        # gradient into hidden activations
        gA = np.outer(grad_out, self.W2)  # (B, H)
        gZ = gA * self.act_grad(Z)        # (B, H)
        gW1 = gZ.T.dot(X) / B             # (H, D)
        gb1 = gZ.sum(axis=0) / B          # (H,)
        return {'W1': gW1, 'b1': gb1, 'W2': gW2, 'b2': gb2}

    def get_params_vector(self):
        # flatten parameters into a single vector
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), np.array([self.b2])])

    def set_params_vector(self, vec):
        # set params from vector (in-place)
        h, d = self.W1.shape
        w1_size = h * d
        self.W1 = vec[:w1_size].reshape(self.W1.shape)
        idx = w1_size
        self.b1 = vec[idx:idx+h]
        idx += h
        self.W2 = vec[idx:idx+h]
        idx += h
        self.b2 = float(vec[idx])

    def get_grad_vector(self, grads):
        return np.concatenate([grads['W1'].ravel(), grads['b1'].ravel(), grads['W2'].ravel(), np.array([grads['b2']])])

    def apply_update_from_vector(self, update_vec, lr):
        # update_vec assumed to be gradient vector (already averaged)
        params = self.get_params_vector()
        params = params - lr * update_vec
        self.set_params_vector(params)

# ------------------ data helpers ------------------
def bucket_locations(df, col, rare_q=0.1, attract_q=0.9):
    """
    Convert location ID into categorical buckets (rare, normal, attractive)
    based on frequency of occurrence in the dataset.

    Args:
        df: DataFrame
        col: column name (e.g., 'PULocationID' or 'DOLocationID')
        rare_q: quantile threshold for "rare"
        attract_q: quantile threshold for "attractive"

    Returns:
        df with new categorical column (col + '_bucket')
    """
    counts = df[col].value_counts(normalize=True)  # frequency
    q_rare = counts.quantile(rare_q)
    q_attract = counts.quantile(attract_q)

    def label_func(x):
        freq = counts.get(x, 0)
        if freq <= q_rare:
            return "rare"
        elif freq >= q_attract:
            return "attractive"
        else:
            return "normal"

    new_col = col + "_bucket"
    df[new_col] = df[col].map(label_func).fillna("rare")
    return df

def bucket_passenger_count(df, col="passenger_count"):
    """
    Bucket passenger_count into 'single', 'normal', 'big_group'.
    """
    def label_func(x):
        if pd.isna(x):
            return "other"   # default bucket
        if x == 1:
            return "single"
        elif 2 == x:
            return "double"
        else:
            return "other"

    new_col = col + "_bucket"
    df[new_col] = df[col].apply(label_func)

    # one-hot encode
    pc_dummies = pd.get_dummies(df[new_col], prefix="passenger")
    df = pd.concat([df, pc_dummies], axis=1)

    # drop original
    df = df.drop(columns=[col, new_col])
    return df

def bucket_ratecode(df, col="RatecodeID"):
    """
    Collapse RatecodeID into two groups:
      - 1.0 (standard rate)
      - other (everything else)
    """
    df["RatecodeID_1"] = (df[col] == 1.0).astype(int)
    df["RatecodeID_other"] = (df[col] != 1.0).astype(int)
    
    # drop original column
    df = df.drop(columns=[col])
    return df

def bucket_payment_type(df, col="payment_type"):
    """
    Collapse payment_type into:
      - card (1)
      - cash (2)
      - other (3,4,...)
    """
    df["payment_1"] = (df[col] == 1).astype(int)
    df["payment_2"] = (df[col] == 2).astype(int)
    df["payment_other"] = (~df[col].isin([1, 2])).astype(int)
    
    return df.drop(columns=[col])

def save_and_print_stats(stats, filename="stats.csv"):
    rows = []
    for col, (mean, std) in stats.items():
        if col == "__target__":
            col = "target (total_amount)"
        rows.append({"Feature": col, "Mean": float(mean), "Std": float(std)})
    df = pd.DataFrame(rows).set_index("Feature")
    pd.DataFrame(rows, columns=["Feature", "Mean", "Std"]).to_csv(filename, index=False)
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def preprocess_and_split(df, features, target_col, seed=42):
    # drop na
    df = df.dropna(subset=features + [target_col]).copy()

    # filter extreme fares (keep 5%–95%)
    with timed_step(logger, rank, "drop 5% highest and lowest"):
        lower = df[target_col].quantile(0.05)
        upper = df[target_col].quantile(0.95)
        df = df[(df[target_col] >= lower) & (df[target_col] <= upper)].copy()
        

    # filter use quantile trimming for trip_distance data
    with timed_step(logger, rank, "bad GPS logs, missing, or too far"):
        # remove near-zero distances , filter to show only > 0.1
        df = df[df["trip_distance"] > 0.1]
        # remove unrealistic long trips, filter to show only <100
        df = df[df["trip_distance"] < 100]
        # other option:
        # low, high = df["trip_distance"].quantile([0.01, 0.99])
        # df = df[(df["trip_distance"] > low) & (df["trip_distance"] < high)]
    
    #checking after drop invalid data to show length of data
    logger.info(f'Length After Drop: ${len(df)}')

    # parse datetime fields if present and create simple features
    with timed_step(logger, rank, "create simple features"):
        format_string = "%m/%d/%Y %I:%M:%S %p"
        with timed_step(logger, rank, "create simple features"):
            df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'], format=format_string, errors='coerce')
            df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'], format=format_string, errors='coerce')
            df['trip_duration_minutes'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60.0
        
        # Filter out trips > 300 minutes
        df = df[(df['trip_duration_minutes'] > 0) & (df['trip_duration_minutes'] <= 300)].copy()

        with timed_step(logger, rank, "Bucket pickup_hour"):
            df['pickup_hour'] = df['pickup_dt'].dt.hour
            # Bucket pickup_hour
            def demand_bucket(hour):
                if 14 <= hour <= 18:
                    return "high"
                elif 8 <= hour <= 13 or 19 <= hour <= 22:
                    return "normal"
                else:
                    return "low"

            df['pickup_hour_bucket'] = df['pickup_hour'].apply(demand_bucket)

            # One-hot encode pickup_hour_bucket
            pickup_dummies = pd.get_dummies(df['pickup_hour_bucket'], prefix="pickup_hour")
            # logger.info(f'pickup_dummies: ${pickup_dummies}')
            df = pd.concat([df, pickup_dummies], axis=1)
        tmp_dt_features = ['tpep_pickup_datetime','tpep_dropoff_datetime','pickup_dt','dropoff_dt','pickup_hour','pickup_hour_bucket']
        df.drop(columns=tmp_dt_features, axis=1, inplace=True)

    with timed_step(logger, rank, "One-hot encode RatecodeID & payment_type"):
        df = bucket_ratecode(df)
        df = bucket_payment_type(df)

    with timed_step(logger, rank, "Bucket pickup and dropoff locations"):
        # Bucket pickup and dropoff locations
        df = bucket_locations(df, "PULocationID")
        df = bucket_locations(df, "DOLocationID")
        # One-hot encode the buckets
        pu_dummies = pd.get_dummies(df["PULocationID_bucket"], prefix="pickup_loc")
        do_dummies = pd.get_dummies(df["DOLocationID_bucket"], prefix="dropoff_loc")

        df = pd.concat([df, pu_dummies, do_dummies], axis=1)

        # Drop the original numeric IDs
        df = df.drop(columns=["PULocationID", "DOLocationID","PULocationID_bucket", "DOLocationID_bucket"])

    with timed_step(logger, rank, "bucket passenger_count"):
        # bucket passenger_count
        df = bucket_passenger_count(df)
    
    with timed_step(logger, rank, "train/test split"):
        # train/test split using a global RNG sequence (so all processes agree)
        rng = np.random.RandomState(seed)
        N = len(df)
        perm = rng.permutation(N)
        n_train = int(0.7 * N)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        return train, test


def normalize_train_test(train_df, test_df):
    # compute mean/std on train
    stats = {}
    X_cols = [c for c in train_df.columns if c != 'total_amount' and c != 'total amount']
    # unify target col name
    target = 'total_amount' if 'total_amount' in train_df.columns else 'total amount'
    for c in X_cols:
        mean = train_df[c].mean()
        std = train_df[c].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        train_df[c] = (train_df[c] - mean) / std
        test_df[c] = (test_df[c] - mean) / std
        stats[c] = (mean, std)
    return train_df, test_df, stats, X_cols, target

# ------------------ training routine ------------------

def train_mpi(args):
    # Rank 0 loads the CSV path and broadcasts nothing — each rank reads the file itself to comply with dataset locality requirement
    if rank == 0:
        logger.info(f"MPI world size: {size}")
        logger.info(f"Loading dataset from: ${args.data}")
    # Each process reads the CSV and keeps rows where index % size == rank
    features_requested = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
        'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type', 'extra'
    ]
    target_col='total_amount'
    focus_columns = features_requested.append(target_col)

    with timed_step(logger, rank, "reads the CSV and keeps rows by rank"):
        df = pd.read_csv(args.data, usecols=focus_columns)
        logger.info(f'Length Global: ${len(df)}')
        # number of rank = portion of data divided to each process, e.g. 1ml data row, with size = 4, each process handle 250k data
        df_local = df.iloc[np.where((np.arange(len(df)) % size) == rank)[0]].reset_index(drop=True)
        logger.info(f'Length Local: ${len(df_local)}')
        # logger.info(df_local.head(5))

    with timed_step(logger, rank, "preprocess_and_split"):
        train_local, test_local  = preprocess_and_split(df_local, features_requested, target_col=target_col, seed=args.seed)
        logger.info(f'Length train_local: ${len(train_local)} test_local: ${len(test_local)}')

    # normalize using training set global stats: compute on rank 0 and broadcast to others
    # To avoid sending full training set, compute stats on rank 0 and broadcast means/stds
    if rank == 0:
        train_for_stats = train_local.copy()
        # train_for_stats.to_csv("sample_data/train_for_stats.csv", index=False)
    else:
        train_for_stats = None
    # We'll construct stats on rank 0 and broadcast via pickleable object
    if rank == 0:
        with timed_step(logger, rank, "normalize_train_test"):
            train_norm, test_norm, stats, X_cols, target = normalize_train_test(train_for_stats.copy(), test_local.copy())
            logger.info(f'\n{save_and_print_stats(stats)}')
    else:
        train_norm = None
        test_norm = None
        stats = None
        X_cols = None
        target = None

    # broadcast stats, X_cols, target to all ranks
    with timed_step(logger, rank, "broadcast stats, X_cols, target to all ranks"):
        stats = comm.bcast(stats, root=0)
        X_cols = comm.bcast(X_cols, root=0)
        target = comm.bcast(target, root=0)

    # Now apply normalization parameters to local partitions
    with timed_step(logger, rank, "apply normalization parameters to local partitions"):
        for c, (mean, std) in stats.items():
            if c in train_local.columns:
                train_local[c] = (train_local[c] - mean) / std
            if c in test_local.columns:
                test_local[c] = (test_local[c] - mean) / std

    # Prepare local arrays
    with timed_step(logger, rank, "Prepare local arrays"):
        X_train = train_local[X_cols].values.astype(np.float64)
        y_train = train_local[target].values.astype(np.float64)
        X_test = test_local[X_cols].values.astype(np.float64)
        y_test = test_local[target].values.astype(np.float64)

        N_train_local = len(X_train)
        logger.info(f'N_train_local {N_train_local}')
        N_train_global = comm.allreduce(N_train_local, op=MPI.SUM)
        if rank == 0:
            logger.info(f"Global train size: {N_train_global}, local sizes: {N_train_local}")

    # initialize model
    model = OneHiddenNN(input_dim=len(X_cols), hidden_dim=args.hidden, activation=args.activation, seed=args.seed + rank)

    with timed_step(logger, rank, "========Training Loop====="):
        # training loop
        history = []
        t0 = time.perf_counter()
        total_iters = 0
        rng = np.random.RandomState(args.seed + rank * 13)
        sync_every = 20  # only sync every 10 batches
        for epoch in range(args.epochs):
            # shuffle local indices
            perm = rng.permutation(N_train_local)

            # compute local and global batch counts
            n_batches_local = int(np.ceil(N_train_local / args.batch_size))
            n_batches_global = comm.allreduce(n_batches_local, op=MPI.MAX)

            if rank == 0:
                logger.info(f"========== Epoch {epoch} START (max {n_batches_global} batches)")
                # safety check
                if psutil.cpu_percent(interval=2) > 90:
                    print("⚠️ CPU too high, aborting all ranks.")
                    comm.Abort(1)
            for b in range(n_batches_global):
                start = b * args.batch_size
                end = start + args.batch_size

                if start < N_train_local:
                    # real batch
                    batch_idx = perm[start:end]
                    Xb, yb = X_train[batch_idx], y_train[batch_idx]

                    # forward
                    ypred, cache = model.forward(Xb)

                    # compute local loss and grad w.r.t outputs
                    local_loss, grad_out = mse_loss_and_grad(ypred, yb)

                    # compute grads w.r.t params
                    grads = model.backward(cache, grad_out)
                    grad_vec = model.get_grad_vector(grads)
                    B_local = Xb.shape[0]
                else:
                    # dummy batch → contribute zeros
                    grad_vec = np.zeros_like(model.get_params_vector())
                    B_local = 0

                # weighted gradient aggregation (important!)
                weighted_grad = grad_vec * B_local
                total_weighted_grad = np.zeros_like(weighted_grad)

                if total_iters % sync_every == 0:
                    # with timed_step(logger, rank, "Allreduce weighted gradient"):
                    comm.Allreduce(weighted_grad, total_weighted_grad, op=MPI.SUM)
                
                # total batch size across all ranks
                global_B = comm.allreduce(B_local, op=MPI.SUM)

                if global_B > 0:
                    total_grad = total_weighted_grad / float(global_B)
                    # apply update
                    model.apply_update_from_vector(total_grad, args.lr)

                total_iters += 1

                # optionally compute global loss every print_every iters
                if total_iters % args.print_every == 0:
                    # local SSE on this rank
                    if N_train_local > 0:
                        ypred_full_local, _ = model.forward(X_train)
                        local_sse = np.sum((ypred_full_local - y_train) ** 2) * 0.5
                    else:
                        local_sse = 0.0

                    # reduce to get global loss
                    global_sse = comm.allreduce(local_sse, op=MPI.SUM)

                    # average loss per sample
                    Rtheta = global_sse / N_train_global
                    history.append((total_iters, Rtheta))

                    # if rank == 0:
                    logger.info(f"Iter {total_iters:6d}, epoch {epoch}, R(θ)={Rtheta:.6f}, R_local(θ)={local_sse:.6f}")

    # compute final RMSE on train and test in parallel
    # local contributions
    with timed_step(logger, rank, "compute final RMSE on train and test in parallel"):
        t1 = time.perf_counter()
        train_time = t1 - t0

        if N_train_local > 0:
            ypred_train_local, _ = model.forward(X_train)
            sse_train_local = np.sum((ypred_train_local - y_train)**2)
        else:
            sse_train_local = 0.0
        sse_train_global = comm.allreduce(sse_train_local, op=MPI.SUM)
        rmse_train = np.sqrt(sse_train_global / max(1, N_train_global))

        N_test_local = len(X_test)
        N_test_global = comm.allreduce(N_test_local, op=MPI.SUM)
        if N_test_local > 0:
            ypred_test_local, _ = model.forward(X_test)
            sse_test_local = np.sum((ypred_test_local - y_test)**2)
        else:
            sse_test_local = 0.0
        sse_test_global = comm.allreduce(sse_test_local, op=MPI.SUM)
        rmse_test = np.sqrt(sse_test_global / max(1, N_test_global))
        # rank 0 prints results
        if rank == 0:
            logger.info("\nTraining finished")
            logger.info(f"Epochs: {args.epochs}, total iters: {total_iters}")
            logger.info(f"Training time (s): {train_time:.3f}")
            logger.info(f"RMSE train: {rmse_train:.6f}")
            logger.info(f"RMSE test:  {rmse_test:.6f}")
            # optionally save model
            if args.save_model:
                params = model.get_params_vector()
                np.savez(args.save_model, params=params, X_cols=X_cols, stats=stats)
                logger.info(f"Saved model to {args.save_model}")

    # collect a small sample from the global test set
    sample_size = min(3, len(X_test))  # take up to 3 from local test
    if sample_size > 0:
        sample_X = X_test[:sample_size]
        sample_y = y_test[:sample_size]
        sample_pred, _ = model.forward(sample_X)

        # print nicely
        logger.info("\nSample test predictions:")
        for i in range(sample_size):
            features = sample_X[i]
            logger.info(f"Features: {features}")
            logger.info(f" Predicted total_amount: {sample_pred[i]:.2f}")
            logger.info(f" Actual total_amount:    {sample_y[i]:.2f}\n")

    # return history for possible further processing
    return history

# ------------------ command-line interface ------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='Path to nytaxi2022.csv')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--activation', type=str, choices=list(ACTIVATIONS.keys()), default='relu')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--print-every', type=int, default=250)
    p.add_argument('--save-model', type=str, default='')
    args = p.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # small validation: require mpi4py
    try:
        import mpi4py
    except Exception:
        if rank == 0:
            logger.info('mpi4py not installed. Install with pip install mpi4py')
        raise

    # run training
    # logger.info(args)
    with timed_step(logger, rank, "run training"):
        hist = train_mpi(args)

    # rank 0 can save history to a CSV if desired
    if rank == 0 and len(hist) > 0:
        import csv
        with open('training_history.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'Rtheta'])
            for it, val in hist:
                writer.writerow([it, val])
        logger.info('Saved training_history.csv')
