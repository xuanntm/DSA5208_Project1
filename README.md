# DSA5208_Project1

## 1) Create environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Create data folder structure
```bash
python 00_create_directory.py
```

> Recommend to keep raw data in data/input/ folder  
> This is the suggested structure
```
.
├── README.md
├── 00_create_directory.py
├── 01_preprocess_data.py
├── 02_data_split.py
├── 03_MPI_SGD_NN_train_v1.py
├── requirements.txt
├── logs/
├── docs/
├── data/
│   ├── input/
│   │   ├── nytaxi2022.csv
│   └── output/
│       ├── cleanup_data/
│       │   └── nytaxi2022_cleaned.csv
│       ├── model/
│       │   └── 20250921_092524.npz
│       ├── split_data/
│       │    ├── to_[N:number of processes]
│       │        └── part_[i:0-> N-1].csv
│       ├── training/
│           └── history_YYYYMMDD_HHMMSS.csv
```

## 📝 Notes
- `data/input/` → Put your raw CSV file here.  
- `data/output/cleanup_data/` → Stores cleaned and standardized datasets.    
- `data/output/split_data/` → Holds split datasets by the number of process. 
- `data/output/model/` → Contains trained models result.
- `data/output/training/` → Contains trained history result.

This structure helps keep raw data separate from processed outputs and models, making the workflow cleaner and reproducible.


## 3) Cleanup, and normalize data
```bash
python 01_preprocess_data.py --input_path data/input/nytaxi2022.csv --output_path data/output/cleanup_data/nytaxi2022_cleaned.csv
```

## 4) Split cleanup data for number of processes
```bash
python 02_data_split.py --input-file-path data/output/cleanup_data/nytaxi2022_cleaned.csv --output-folder data/output/split_data --number-process 8
```

## 5) install MPI for macbook
```bash
brew install open-mpi
```

## 6) Training with MPI on single computer
> Run with default parameters
```bash
mpiexec -n 4 python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data
```
> Run with custom parameters
```bash
mpiexec -n 3 python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --epochs 1 --batch-size 512 --hidden 64 --lr 0.002 --activation relu
```
```
**Note**: with fulldata set ~ 39ml data; one Laptop (Memory of 18GB) cannot run whole dataset.
For the training with MPI on single computer; can split data into multiple set; and only run few of them. Recomemdation: Split data into 8 set; and run MPI with 3 set (3 process) in 1 Laptop.
```


## 7) Config for multiple computers - For MACBOOK
```text
- enable remote access on second laptop
- try to connect from first computer to second computer
- Sample: ssh XuanNguyen@192.168.1.9 # UserName and local IP (wifi) ??? LAN network
- Run the same step on second laptop after clone source code from github
1) Create environment
2) Create data folder structure
- Copy split file from first computer to second compute (depend on where you keep the source code)
- Example: scp -r data/output/split_data/to_20/ username@remote_host:{PROJECT_DIRECTORY}/data/output/split_data/to_20/
```

## 8) Training with MPI on multiple computers
> Each computer will have to run from **Step 1 to 5 with same config** (number of process)  
> Total Process (N) = N_FIRST + N_SECOND
```bash
mpiexec -host {{LOCAL_IP}}:{{N_FIRST}} venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data : \
        -host XuanNguyen@{{REMOTE_IP}}:{{N_SECOND}} {{PROJECT_DIRECTORY}}/venv/bin/python {{PROJECT_DIRECTORY}}/03_MPI_SGD_NN_train_v1.py --data {{PROJECT_DIRECTORY}}/data/output/split_data
```
Sample :
```
mpiexec -host 192.168.1.4:5 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data : \
        -host XuanNguyen@192.168.1.9:3 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data


mpiexec -host 192.168.1.4:5 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --sync-every 50 : \
        -host XuanNguyen@192.168.1.9:3 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --sync-every 50

```