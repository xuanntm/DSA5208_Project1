import logging
from mpi4py import MPI

def setup_logger():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"rank{rank}.log", mode="w")
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - [Rank {rank}] - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        if rank == 0:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)
    return logger
