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
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import logging
import warnings
warnings.filterwarnings("ignore", message="Could not infer format")


# Configure the basic logging settings
# The format string includes '%(asctime)s' to automatically add the timestamp
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logger.info)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# logging.basicConfig(
#     format=f'%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s',
#     level=logger.info
# )

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

logger.info("Training started")

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
        self.W1 = rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / max(1, input_dim))
        self.b1 = np.zeros((hidden_dim,))
        # output layer weights
        self.W2 = rng.randn(hidden_dim) * np.sqrt(2.0 / max(1, hidden_dim))
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

def preprocess_and_split(df, features, target_col, seed=42):
    # drop na
    # print('===START.1')
    # print(features)
    df = df.dropna(subset=features + [target_col]).copy()

    # filter extreme fares (keep 5%–95%)
    # target = 'total_amount' if 'total_amount' in df.columns else 'total amount'
    # lower = df[target_col].quantile(0.05)
    # upper = df[target_col].quantile(0.95)
    # df = df[(df[target_col] >= lower) & (df[target_col] <= upper)].copy()

    # print('===END.1')
    # print(df.head(2))
    # parse datetime fields if present and create simple features
    if 'tpep_pickup_datetime' in df.columns:
        df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    else:
        df['pickup_dt'] = pd.NaT

    if 'tpep_dropoff_datetime' in df.columns:
        df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    else:
        df['dropoff_dt'] = pd.NaT

    # duration (seconds) and pickup hour
    df['trip_duration'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds()
    # Trip duration in minutes
    df['trip_duration_minutes'] = df['trip_duration'] / 60.0

    # Filter out trips > 300 minutes
    # df = df[df['trip_duration_minutes'] <= 300].copy()

    df['pickup_hour'] = df['pickup_dt'].dt.hour
    # Bucket pickup_hour
    def demand_bucket(hour):
        if 2 <= hour < 6:
            return "low"
        elif 6 <= hour < 16:
            return "normal"
        else:
            return "high"

    df['pickup_hour_bucket'] = df['pickup_hour'].apply(demand_bucket)

    # One-hot encode pickup_hour_bucket
    pickup_dummies = pd.get_dummies(df['pickup_hour_bucket'], prefix="pickup_hour")
    df = pd.concat([df, pickup_dummies], axis=1)
    
    # Drop old columns no longer needed
    df = df.drop(columns=['pickup_hour', 'trip_duration', 'pickup_hour_bucket'])

    # keep features present only
    cols = []
    for f in features:
        if f in df.columns:
            cols.append(f)
    # Add derived features
    # cols += ['trip_duration', 'pickup_hour']
    cols += ['trip_duration_minutes', 'pickup_hour_bucket']

    # We should drop the origin feature and keep the new features
    columns_to_drop = ['tpep_pickup_datetime', 'tpep_dropoff_datetime','dropoff_dt','pickup_dt']
    # Drop the specified columns.
    # axis=1 tells pandas to drop columns, not rows.
    # inplace=True modifies the DataFrame directly, without creating a new one.
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    logger.info("save data_step1")
    # df.to_csv("tmp/data_step1.csv", index=False)
    

    # print('=====cols')
    # print(cols)
    # For categorical columns, do label encoding (simple). One-hot would explode for PULocationID.
    # cat_cols = []
    # for c in ['RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type']:
    #     if c in df.columns:
    #         df[c] = df[c].astype('category').cat.codes.astype(float)
    #         cols.append(c)
    #         cat_cols.append(c)

    # 1. One-hot encode RatecodeID & payment_type
    onehot_features = ["RatecodeID", "payment_type"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    onehot_encoded = ohe.fit_transform(df[onehot_features])
    onehot_df = pd.DataFrame(
        onehot_encoded,
        columns=ohe.get_feature_names_out(onehot_features),
        index=df.index
    )

    # 2. Label encode PULocationID & DOLocationID
    label_features = ["PULocationID", "DOLocationID"]
    label_encoders = {}
    for col in label_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # store encoder for inverse transform if needed

    # -------------------------------
    # Combine processed features
    # -------------------------------
    # Drop original one-hot encoded columns & merge
    df = df.drop(columns=onehot_features)
    df = pd.concat([df, onehot_df], axis=1)
    logger.info("save data_step2")
    # df.to_csv("tmp/data_step2.csv", index=False)


    # numerical fill and selection
    # df = df[cols + [target_col]].copy()
    # drop rows with NaN (e.g. bad datetimes)
    # df = df.dropna()

    # print(df.head(2))

    newFeatures = df.columns

    # train/test split using a global RNG sequence (so all processes agree)
    rng = np.random.RandomState(seed)
    N = len(df)
    perm = rng.permutation(N)
    n_train = int(0.7 * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    return train, test, newFeatures


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
    df = pd.read_csv(args.data)
    logger.info("read_csv Successful")
    # df_local = df.iloc[np.where((np.arange(len(df)) % size) == rank)[0]].reset_index(drop=True)
    # print(df_local.head(5))

    features_requested = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
        'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type', 'extra'
    ]
    # the preprocess function will handle alternate column names
    train_df_global, test_df_global, newFeatures  = preprocess_and_split(df, features_requested, target_col='total_amount', seed=args.seed)
    logger.info("preprocess_and_split completed")
    logger.info(f"newFeatures: ${newFeatures}")

    # Now each process selects its local slice from the global train/test (so partitions are consistent)
    # Note: we want the dataset to be distributed among processes; to do that, we partition the global train and test by row index across processes
    def local_partition(global_df):
        N = len(global_df)
        idxs = np.arange(N)
        local_mask = (idxs % size) == rank
        logger.info(f"local_mask:${local_mask}")
        return global_df.iloc[local_mask].reset_index(drop=True)
    
    logger.info("Start training local")
    train_local = local_partition(train_df_global)
    logger.info("End training local")
    test_local = local_partition(test_df_global)

    # normalize using training set global stats: compute on rank 0 and broadcast to others
    # To avoid sending full training set, compute stats on rank 0 and broadcast means/stds
    if rank == 0:
        train_for_stats = train_df_global.copy()
    else:
        train_for_stats = None
    # We'll construct stats on rank 0 and broadcast via pickleable object
    if rank == 0:
        logger.info("Start normalize_train_test")
        train_norm, test_norm, stats, X_cols, target = normalize_train_test(train_for_stats.copy(), test_df_global.copy())
        logger.info("End normalize_train_test")
    else:
        train_norm = None
        test_norm = None
        stats = None
        X_cols = None
        target = None

    # logger.info(f'stats:${stats}')
    # logger.info(f'X_cols:${X_cols}')

    # broadcast stats, X_cols, target to all ranks
    stats = comm.bcast(stats, root=0)
    X_cols = comm.bcast(X_cols, root=0)
    target = comm.bcast(target, root=0)

    # Now apply normalization parameters to local partitions
    for c, (mean, std) in stats.items():
        if c in train_local.columns:
            train_local[c] = (train_local[c] - mean) / std
        if c in test_local.columns:
            test_local[c] = (test_local[c] - mean) / std

    # Prepare local arrays
    X_train = train_local[X_cols].values.astype(np.float64)
    y_train = train_local[target].values.astype(np.float64)
    X_test = test_local[X_cols].values.astype(np.float64)
    y_test = test_local[target].values.astype(np.float64)

    N_train_local = len(X_train)
    N_train_global = comm.allreduce(N_train_local, op=MPI.SUM)
    if rank == 0:
        logger.info(f"Global train size: {N_train_global}, local sizes: {N_train_local}")

    # initialize model
    model = OneHiddenNN(input_dim=len(X_cols), hidden_dim=args.hidden, activation=args.activation, seed=args.seed + rank)

    # training loop
    history = []
    t0 = time.perf_counter()
    total_iters = 0
    rng = np.random.RandomState(args.seed + rank * 13)

    for epoch in range(args.epochs):
        # shuffle local indices
        perm = rng.permutation(N_train_local)
        # iterate mini-batches on local data
        for start in range(0, N_train_local, args.batch_size):
            batch_idx = perm[start:start+args.batch_size]
            if len(batch_idx) == 0:
                continue
            Xb = X_train[batch_idx]
            yb = y_train[batch_idx]
            # forward
            ypred, cache = model.forward(Xb)
            # compute local loss and grad w.r.t outputs
            local_loss, grad_out = mse_loss_and_grad(ypred, yb)
            # compute grads w.r.t params (averaged over local batch inside backward)
            grads = model.backward(cache, grad_out)
            grad_vec = model.get_grad_vector(grads)
            # Now aggregate gradients across processes (average): sum then divide
            total_grad = np.zeros_like(grad_vec)
            comm.Allreduce(grad_vec, total_grad, op=MPI.SUM)
            total_grad /= size  # average across ranks
            # apply update locally
            model.apply_update_from_vector(total_grad, args.lr)

            total_iters += 1
            # optionally compute global loss every print_every iters
            if total_iters % args.print_every == 0:
                # compute local sum-of-squared-errors
                ypred_full_local, _ = model.forward(X_train)
                local_sse = np.sum((ypred_full_local - y_train)**2) * 0.5
                # reduce to get global loss
                global_sse = comm.allreduce(local_sse, op=MPI.SUM)
                # compute average loss per sample
                Rtheta = global_sse / N_train_global
                history.append((total_iters, Rtheta))
                if rank == 0:
                    logger.info(f"Iter {total_iters:6d}, epoch {epoch}, R(θ)={Rtheta:.6f}")

    t1 = time.perf_counter()
    train_time = t1 - t0

    # compute final RMSE on train and test in parallel
    # local contributions
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
        sample_size = min(2, len(X_test))  # take up to 2 from local test
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
    p.add_argument('--print-every', type=int, default=100)
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
    logger.info("Start train_mpi")
    logger.info(args)
    hist = train_mpi(args)
    logger.info("End train_mpi")

    # rank 0 can save history to a CSV if desired
    if rank == 0 and len(hist) > 0:
        import csv
        with open('training_history.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'Rtheta'])
            for it, val in hist:
                writer.writerow([it, val])
        logger.info('Saved training_history.csv')
