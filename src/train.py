import time
import numpy as np
import pandas as pd
from mpi4py import MPI

from model import OneHiddenNN
from data import preprocess_and_split, normalize_train_test
from utils import mse_loss_and_grad

def train_mpi(args, logger):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        logger.info(f"MPI world size: {size}")
        logger.info(f"Loading dataset from: {args.data}")

    # Each process reads the CSV
    df = pd.read_csv(args.data)

    # Feature columns requested
    features_requested = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "passenger_count", "trip_distance",
        "RatecodeID", "PULocationID", "DOLocationID",
        "payment_type", "extra"
    ]

    # Preprocess globally (each rank runs same function)
    train_df_global, test_df_global = preprocess_and_split(
        df, features_requested, target_col="total_amount", seed=args.seed
    )

    # Partition across processes
    def local_partition(global_df):
        idxs = np.arange(len(global_df))
        local_mask = (idxs % size) == rank
        return global_df.iloc[local_mask].reset_index(drop=True)

    train_local = local_partition(train_df_global)
    test_local = local_partition(test_df_global)

    # Normalize (compute stats on rank 0 and broadcast)
    if rank == 0:
        train_norm, test_norm, stats, X_cols, target = normalize_train_test(
            train_df_global.copy(), test_df_global.copy()
        )
    else:
        stats, X_cols, target = None, None, None

    stats = comm.bcast(stats, root=0)
    X_cols = comm.bcast(X_cols, root=0)
    target = comm.bcast(target, root=0)

    # Apply normalization params to local partitions
    for c, (mean, std) in stats.items():
        if c == "__target__":
            continue
        if c in train_local.columns:
            train_local[c] = (train_local[c] - mean) / std
        if c in test_local.columns:
            test_local[c] = (test_local[c] - mean) / std

    # Normalize target locally
    y_mean, y_std = stats["__target__"]
    train_local[target] = (train_local[target] - y_mean) / y_std
    test_local[target] = (test_local[target] - y_mean) / y_std

    # Prepare arrays
    X_train = train_local[X_cols].values.astype(np.float64)
    y_train = train_local[target].values.astype(np.float64)
    X_test = test_local[X_cols].values.astype(np.float64)
    y_test = test_local[target].values.astype(np.float64)

    N_train_local = len(X_train)
    N_train_global = comm.allreduce(N_train_local, op=MPI.SUM)
    if rank == 0:
        logger.info(f"Global train size: {N_train_global}, local size: {N_train_local}")

    # Initialize model
    model = OneHiddenNN(input_dim=len(X_cols),
                        hidden_dim=args.hidden,
                        activation=args.activation,
                        seed=args.seed + rank)

    # Training loop
    history = []
    t0 = time.perf_counter()
    total_iters = 0
    rng = np.random.RandomState(args.seed + rank * 13)

    for epoch in range(args.epochs):
        perm = rng.permutation(N_train_local)
        for start in range(0, N_train_local, args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            if len(batch_idx) == 0:
                continue
            Xb, yb = X_train[batch_idx], y_train[batch_idx]

            ypred, cache = model.forward(Xb)
            local_loss, grad_out = mse_loss_and_grad(ypred, yb)
            grads = model.backward(cache, grad_out)
            grad_vec = model.get_grad_vector(grads)

            total_grad = np.zeros_like(grad_vec)
            comm.Allreduce(grad_vec, total_grad, op=MPI.SUM)
            total_grad /= size

            model.apply_update_from_vector(total_grad, args.lr)

            total_iters += 1
            if total_iters % args.print_every == 0:
                ypred_full_local, _ = model.forward(X_train)
                local_sse = np.sum((ypred_full_local - y_train) ** 2) * 0.5
                global_sse = comm.allreduce(local_sse, op=MPI.SUM)
                Rtheta = global_sse / N_train_global
                history.append((total_iters, Rtheta))
                if rank == 0:
                    logger.info(f"Iter {total_iters}, epoch {epoch}, R(θ)={Rtheta:.6f}")

    t1 = time.perf_counter()
    train_time = t1 - t0

    # Final RMSE (denormalized)
    def global_rmse(X, y):
        N_local = len(X)
        N_global = comm.allreduce(N_local, op=MPI.SUM)
        if N_local > 0:
            ypred_local, _ = model.forward(X)
            sse_local = np.sum((ypred_local - y) ** 2)
        else:
            sse_local = 0.0
        sse_global = comm.allreduce(sse_local, op=MPI.SUM)
        rmse_norm = np.sqrt(sse_global / max(1, N_global))
        return rmse_norm * y_std

    rmse_train = global_rmse(X_train, y_train)
    rmse_test = global_rmse(X_test, y_test)

    if rank == 0:
        logger.info("Training finished")
        logger.info(f"Epochs: {args.epochs}, total iters: {total_iters}")
        logger.info(f"Training time (s): {train_time:.3f}")
        logger.info(f"RMSE train: {rmse_train:.6f}")
        logger.info(f"RMSE test:  {rmse_test:.6f}")

        if args.save_model:
            params = model.get_params_vector()
            np.savez(args.save_model, params=params, X_cols=X_cols, stats=stats)
            logger.info(f"Saved model to {args.save_model}")

    return history
