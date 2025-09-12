# DSA5208_Project1

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt


mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_1000.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.01 --activation relu

## good result (Best)
mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_100000.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.001 --activation relu


mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_100000.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.01 --activation relu
## learn slower -> have same result
mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_100000.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.001 --activation relu

mpiexec -n 4 python main.py --data sample_data/nytaxi2022_100000.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.001 --activation relu

mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_1000000.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.001 --activation relu

mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_1000000.csv --epochs 30 --batch-size 128 --hidden 64 --lr 0.001 --activation relu


mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_1000000.csv --epochs 10 --batch-size 512 --hidden 128 --lr 0.005 --activation relu

### Make OS down !!!!!
mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022.csv --epochs 10 --batch-size 512 --hidden 128 --lr 0.001 --activation relu


### 5000000
mpiexec -n 4 python MPI_SGD_NN_train.py --data sample_data/nytaxi2022_5000000.csv --epochs 30 --batch-size 256 --hidden 64 --lr 0.002 --activation relu


--epochs 300 ==> Cham qua khong chiu noi

## Have to install library for macbook M2

sudo chown -R XuanNguyen /opt/homebrew

brew install open-mpi

## How to filter the invalid data ?
1. negative data
2. random data