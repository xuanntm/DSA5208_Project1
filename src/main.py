import argparse
from train import train_mpi
from logger import setup_logger

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--activation', type=str, default='relu',
                   choices=['relu', 'tanh', 'sigmoid'])
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--print-every', type=int, default=100)
    p.add_argument('--save-model', type=str, default='')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Starting training")
    hist = train_mpi(args, logger)
    if logger:
        logger.info("Training finished")
