import argparse
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
import logging
from contextlib import contextmanager
from datetime import datetime


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
fh = logging.FileHandler(f"logs/rank{rank}.log", mode="w")
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

# Open a shared file for logging
# mode = MPI.MODE_WRONLY | MPI.MODE_CREATE
# fh = MPI.File.Open(comm, "shared_logfile.log", mode)
# fh.Set_atomicity(True) # Ensure atomic writes

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


def calculate_sse(model, X, Y, calculate_size=128):
    sse = 0.0
    n_samples = X.shape[0]

    for start in range(0, n_samples, calculate_size):
        end = start + calculate_size
        X_batch = X[start:end]
        Y_batch = Y[start:end]

        # forward pass for this batch
        ypred_batch, _ = model.forward(X_batch)

        # accumulate squared error
        sse += np.sum((ypred_batch - Y_batch) ** 2)

    return sse


# ------------------ neural network ------------------
class OneHiddenNN:
    def __init__(self, input_dim, hidden_dim, activation='relu', seed=1234):
        rng = np.random.RandomState(seed)
        # Xavier init for hidden weights
        self.W1 = rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / max(1, input_dim))
        # self.W1 = rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / max(1, input_dim)) * 0.1
        self.b1 = np.zeros((hidden_dim,))
        # output layer weights
        self.W2 = rng.randn(hidden_dim) * np.sqrt(2.0 / max(1, hidden_dim))
        # self.W2 = rng.randn(hidden_dim) * np.sqrt(2.0 / max(1, hidden_dim)) * 0.1
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

# ------------------ neural network ------------------
def save_and_print_stats(stats, filename="data/output/training/stats.csv"):
    rows = []
    for col, (mean, std) in stats.items():
        if col == "__target__":
            col = "target (total_amount)"
        rows.append({"Feature": col, "Mean": float(mean), "Std": float(std)})
    df = pd.DataFrame(rows).set_index("Feature")
    pd.DataFrame(rows, columns=["Feature", "Mean", "Std"]).to_csv(filename, index=False)
    return df.to_string(float_format=lambda x: f"{x:,.3f}")

def split_for_training(df, seed=42):
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

def normalize_train_test(train_df, test_df, target):
    # compute mean/std on train
    stats = {}
    X_cols = [c for c in train_df.columns if c != target]
    for c in X_cols:
        mean = train_df[c].mean()
        std = train_df[c].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        train_df[c] = (train_df[c] - mean) / std
        test_df[c] = (test_df[c] - mean) / std
        stats[c] = (mean, std)
    return train_df, test_df, stats, X_cols

# ------------------ training routine ------------------

def train_mpi(args):
    # Rank 0 loads the CSV path and broadcasts nothing — each rank reads the file itself to comply with dataset locality requirement
    if rank == 0:
        logger.info(f"MPI world size: {size}")
        logger.info(f"Loading dataset from: ${args.data}")
    target='total_amount'

    # each process will load their data
    # with timed_step(logger, rank, "reads the CSV and keeps rows by rank"):
    file_by_rank = f'{args.data}/to_{size}/part_{rank}.csv'
    # logger.info(f"file_by_rank: {file_by_rank}")
    df = pd.read_csv(file_by_rank)

    # with timed_step(logger, rank, "split_for_training"):
    train_local, test_local  = split_for_training(df, seed=args.seed)
    # logger.info(f'Length train_local: ${len(train_local)} test_local: ${len(test_local)}')

    # normalize using training set global stats: compute on rank 0 and broadcast to others
    # To avoid sending full training set, compute stats on rank 0 and broadcast means/stds
    
    if rank == 0:
        train_for_stats = train_local.copy()
        # with timed_step(logger, rank, "normalize_train_test"):
        train_norm, test_norm, stats, X_cols = normalize_train_test(train_for_stats.copy(), test_local.copy(), target)
        # logger.info(f'\n{save_and_print_stats(stats)}')
    else:
        train_for_stats = None
        train_norm = None
        test_norm = None
        stats = None
        X_cols = None
        target = None
    
    
    # We'll construct stats on rank 0 and broadcast via pickleable object
    # broadcast stats, X_cols, target to all ranks
    # logger.info("broadcast stats, X_cols, target to all ranks")
    stats = comm.bcast(stats, root=0)
    X_cols = comm.bcast(X_cols, root=0)
    target = comm.bcast(target, root=0)

    # Now apply normalization parameters to local partitions
    # logger.info("apply normalization parameters to local partitions")
    for c, (mean, std) in stats.items():
        if c in train_local.columns:
            train_local[c] = (train_local[c] - mean) / std
        if c in test_local.columns:
            test_local[c] = (test_local[c] - mean) / std

    # Prepare local arrays
    # logger.info("Prepare local arrays")
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
        # for Multiple round training
        logger.info(f"===== Number of local train {N_train_local / args.batch_size}")
        for epoch in range(args.epochs):
            perm = rng.permutation(N_train_local)

            if rank == 0:
                logger.info(f"========== Epoch {epoch} START")

            for b, start in enumerate(range(0, N_train_local, args.batch_size)):
                end = start + args.batch_size
                batch_idx = perm[start:end]
                if len(batch_idx) == 0:
                    continue

                Xb, yb = X_train[batch_idx], y_train[batch_idx]

                # forward
                ypred, cache = model.forward(Xb)

                # compute local loss and grad w.r.t outputs
                local_loss, grad_out = mse_loss_and_grad(ypred, yb)

                # compute grads w.r.t params
                grads = model.backward(cache, grad_out)
                grad_vec = model.get_grad_vector(grads)

                # apply local update
                model.apply_update_from_vector(grad_vec, args.lr)

                total_iters += 1

                # 🔹 Hybrid sync: every K batches
                if args.sync_every > 0 and (b + 1) % args.sync_every == 0:
                    param_vec = model.get_params_vector()
                    avg_param = np.zeros_like(param_vec)
                    comm.Allreduce(param_vec, avg_param, op=MPI.SUM)
                    avg_param /= size
                    model.set_params_vector(avg_param)

                # 🔹 Print R(θ) every N batches (default 100)
                # if total_iters % args.print_every == 0:
                #     if N_train_local > 0:
                #         ypred_full_local, _ = model.forward(X_train)
                #         local_sse = np.sum((ypred_full_local - y_train) ** 2) * 0.5
                #     else:
                #         local_sse = 0.0

                #     global_sse = comm.allreduce(local_sse, op=MPI.SUM)
                #     Rtheta = global_sse / N_train_global
                #     history.append((total_iters, Rtheta))

                #     if rank == 0:
                #         logger.info(f"Iter {total_iters:6d}, epoch {epoch}, R(θ)={Rtheta:.6f}")
                
                
                # 🔹 Print *local* R(θ) every N batches
                if total_iters % args.print_every == 0:
                    # logger.info(f"============total_iters {total_iters}")
                    if N_train_local > 0:
                        # ypred_full_local, _ = model.forward(X_train)
                        # local_sse = np.sum((ypred_full_local - y_train) ** 2) * 0.5
                        local_sse = calculate_sse(model, X_train, y_train, args.calculate_size)
                        Rtheta_local = local_sse / N_train_local
                        logger.info(f"[Rank {rank}] Iter {total_iters:6d}, epoch {epoch}, "
                                    f"local R(θ)={Rtheta_local:.6f}")


            # 🔹 Epoch-end sync (only if sync_every == 0)
            if args.sync_every == 0:
                param_vec = model.get_params_vector()
                avg_param = np.zeros_like(param_vec)
                comm.Allreduce(param_vec, avg_param, op=MPI.SUM)
                avg_param /= size
                model.set_params_vector(avg_param)

            # ---- compute global loss after epoch sync ----
            if N_train_local > 0:
                # ypred_full_local, _ = model.forward(X_train)
                # local_sse = np.sum((ypred_full_local - y_train) ** 2) * 0.5
                local_sse = calculate_sse(model, X_train, y_train, args.calculate_size)
            else:
                local_sse = 0.0

            global_sse = comm.allreduce(local_sse, op=MPI.SUM)
            Rtheta = global_sse / N_train_global
            history.append((total_iters, Rtheta))

            if rank == 0:
                logger.info(f"Epoch {epoch} END, R(θ)={Rtheta:.6f}")

    # compute final RMSE on train and test in parallel
    # local contributions
    with timed_step(logger, rank, "compute final RMSE on train and test in parallel"):
        t1 = time.perf_counter()
        train_time = t1 - t0
        if N_train_local > 0:
            # ypred_train_local, _ = model.forward(X_train)
            # sse_train_local = np.sum((ypred_train_local - y_train)**2)
            sse_train_local = calculate_sse(model, X_train, y_train, args.calculate_size)
        else:
            sse_train_local = 0.0
        sse_train_global = comm.allreduce(sse_train_local, op=MPI.SUM)
        rmse_train = np.sqrt(sse_train_global / max(1, N_train_global))

        N_test_local = len(X_test)
        N_test_global = comm.allreduce(N_test_local, op=MPI.SUM)
        if N_test_local > 0:
            sse_test_local = calculate_sse(model, X_test, y_test, args.calculate_size)
        else:
            sse_test_local = 0.0
        sse_test_global = comm.allreduce(sse_test_local, op=MPI.SUM)
        rmse_test = np.sqrt(sse_test_global / max(1, N_test_global))
        # rank 0 prints results
        if rank == 0:
            logger.info("\nTraining finished")
            logger.info(f"{args}")
            logger.info(f"Epochs: {args.epochs}, total iters: {total_iters}")
            logger.info(f"Training time (s): {train_time:.3f}")
            logger.info(f"RMSE train: {rmse_train:.6f}")
            logger.info(f"RMSE test:  {rmse_test:.6f}")
            # optionally save model
            if args.save_model:
                params = model.get_params_vector()
                np.savez(f"{args.save_model}{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz", params=params, X_cols=X_cols, stats=stats, args=args)
                logger.info(f"Saved model to {args.save_model}")

    # collect a small sample from the global test set
    # sample_size = min(3, len(X_test))  # take up to 3 from local test
    # if sample_size > 0:
    #     sample_X = X_test[:sample_size]
    #     sample_y = y_test[:sample_size]
    #     sample_pred, _ = model.forward(sample_X)

    #     # print nicely
    #     logger.info("\nSample test predictions:")
    #     for i in range(sample_size):
    #         features = sample_X[i]
    #         logger.info(f"Features: {features}")
    #         logger.info(f" Predicted total_amount: {sample_pred[i]:.2f}")
    #         logger.info(f" Actual total_amount:    {sample_y[i]:.2f}\n")

    # return history for possible further processing
    return history

# ------------------ command-line interface ------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='Path to nytaxi2022.csv')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--calculate-size', type=int, default=128)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--activation', type=str, choices=list(ACTIVATIONS.keys()), default='relu')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--sync-every', type=int, default=0)
    p.add_argument('--print-every', type=int, default=2500)
    p.add_argument('--save-model', type=str, default='data/output/model/')
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
        with open(f"data/output/training/history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'Rtheta'])
            for it, val in hist:
                writer.writerow([it, val])
        logger.info('Saved training history.csv')
