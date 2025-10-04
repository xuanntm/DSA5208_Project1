# DSA5208_Project1

## 1) Create environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Create data folder structure (program: 00_create_directory.py)
```bash
python 00_create_directory.py
```
> Recommend to keep raw data in data/input/ folder  
> This is the suggested structure
```
.
├── README.md           #document presents technical guide to config and run project
├── REPORTv2.md         #present theoretical part and results of project
├── 00_create_directory.py      #program to run for creating folder structure
├── 01_preprocess_data.py       #program to pre-process data, explained rationale in REPORTv2.md
├── 02_data_split.py            #program to split data 70%-30% per request
├── 03_MPI_SGD_NN_train_v1.py   #main program to expereince neural network model & MPI
├── 04_MPI_SGD_NN_train_support_single_laptop.py        #testing model to run in single laptop
├── requirements.txt            #requirement
├── z_data_analyze.ipynb        #support material to analyse dataset before preprocess data
├── z_extract_log_to_csv.ipynb  #supporting program to run report and produce chart from logs (output of 03_MPI_SGD_NN_train_v1.py )
├── z_extract_raw.ipynb         #supporting program to extract the small part of whole dataset to test the capacity of a computer 
├── logs/                       #record the output from main program 03_MPI_SGD_NN_train_v1.py
├── docs/                       #record the diagram to present the flow of data and step of program
├── data/                       #contain the dataset input setup and output (not commit to git)
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
│       │      └── history_YYYYMMDD_HHMMSS.csv
├── charts/                     #output from z_extract_log_to_csv.ipynb 
├── report/                     #output from z_extract_log_to_csv.ipynb 
```

## 📝 Notes
- `data/input/` → Put your raw CSV file here.  
- `data/output/cleanup_data/` → Stores cleaned and standardized datasets.    
- `data/output/split_data/` → Holds split datasets by the number of process. 
- `data/output/model/` → Contains trained models result.
- `data/output/training/` → Contains trained history result.

This structure helps keep raw data separate from processed outputs and models, making the workflow cleaner and reproducible.


## 3) Cleanup, and normalize data (program: 01_preprocess_data.py)
```bash
python 01_preprocess_data.py --input_path data/input/nytaxi2022.csv --output_path data/output/cleanup_data/nytaxi2022_cleaned.csv
```

## 4) Split cleanup data for number of processes (program: 02_data_split.py)
```bash
python 02_data_split.py --input-file-path data/output/cleanup_data/nytaxi2022_cleaned.csv --output-folder data/output/split_data --number-process 8
```

## 5) install MPI for macbook
- for macbook
```bash
brew install open-mpi

```

- for windows (optional - following CLI run - basing on Macbook only)

>1) Install Microsoft MPI (MS-MPI)

>- Download and install Microsoft MPI (MS-MPI) Redistributable (64-bit): `MSMpiSetup.exe`

>- After install, mpiexec.exe typically lives at: `C:\Program Files\Microsoft MPI\Bin\mpiexec.exe`

>2) Verify
```powershell
# Test that MPI + Python work together
mpiexec -n 2 py -c "from mpi4py import MPI; comm=MPI.COMM_WORLD; print(f'rank {comm.Get_rank()} of {comm.Get_size()}')"
```
>>> You should see two lines like:
```nginx
rank 0 of 2
rank 1 of 2
```

## 6) Training with MPI on single computer (program: 04_MPI_SGD_NN_train_support_single_laptop)
> Run with default parameters
```bash
mpiexec -n 4 python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data
```
> Run with custom parameters
```bash
mpiexec -n 3 python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --epochs 1 --batch-size 512 --hidden 64 --lr 0.002 --activation relu
```
```text
**Note**: with fulldata set ~ 39ml data; one Laptop (Memory of 18GB) cannot run program: 03_MPI_SGD_NN_train_v1 with whole dataset.
For the purpose of testing model neural network with all dataset, we created a separate:
``` 
**program: 04_MPI_SGD_NN_train_support_single_laptop** 
including **--calculate_sse** ; which will divide data into small set within a batch to load and calculate within each batch, which reduce the demanding of high memory of laptop to store. 
Since the purpose of project is to verify the MPI technology ability to help connect multiple computer to run neural network model with big data. So, --calculate_sse only use for this testing on 1 computer. In the section 7) - MPI, we remove --calculate_sse from main program.
```
```
## 7) Config for multiple computers - For MACBOOK (program: 03_MPI_SGD_NN_train_v1)
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

## 8) Training with MPI on multiple computers (program: 03_MPI_SGD_NN_train_v1)
> Each computer will have to run from **Step 1 to 5 with same config**  
> Total Process (N) = N_FIRST + N_SECOND (N_FIRST = number of process for host/main computer; and N_SECOND = number of process allocated to 2nd computer)
```bash
mpiexec -host {{LOCAL_IP}}:{{N_FIRST}} venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data : \
        -host XuanNguyen@{{REMOTE_IP}}:{{N_SECOND}} {{PROJECT_DIRECTORY}}/venv/bin/python {{PROJECT_DIRECTORY}}/03_MPI_SGD_NN_train_v1.py --data {{PROJECT_DIRECTORY}}/data/output/split_data
```
Default config:
- '--epochs', type=int, default=1
- '--batch-size', type=int, default=1024
- '--hidden', type=int, default=64
- '--lr', type=float, default=0.002
- '--activation', type=str, choices=list(ACTIVATIONS.keys()), default='relu'
- '--seed', type=int, default=123
- '--sync-every', type=int, default=0
- '--print-every', type=int, default=2500

Output will be recorded in #logs; then manually intervention to rename log-rank0 to record result, before start new CLI for different experiment. 


## 9) CLI script for Training/Test (program: 03_MPI_SGD_NN_train_v1)

**3 activations** × **5 batch sizes** at **Process = 8**
```bash
mpiexec -host 192.168.1.7:5 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 128 --print-every 2500 --activation relu : \
        -host XuanNguyen@192.168.1.9:3 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 128 --print-every 100000 --activation relu

mpiexec -host 192.168.1.7:5 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 128 --print-every 2500 --activation sigmoid : \
        -host XuanNguyen@192.168.1.9:3 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 128 --print-every 100000 --activation sigmoid

mpiexec -host 192.168.1.7:5 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 128 --print-every 2500 --activation tanh : \
        -host XuanNguyen@192.168.1.9:3 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 128 --print-every 100000 --activation tanh
```
**different numbers of processes**
```text
Laptop 1 runs 5 processes while Laptop 2 runs 7 processes
```
```bash
mpiexec -host 192.168.1.7:7 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 210 --activation relu : \
        -host XuanNguyen@192.168.1.9:5 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation relu
```
```text
Laptop 1 runs 6 processes while Laptop 2 runs 10 processes
```
```bash
mpiexec -host 192.168.1.7:10 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 160 --activation relu : \
        -host XuanNguyen@192.168.1.9:6 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation relu

```
**Efforts to Improve Results & Performance**

Trial 1: Experience the RMSE and Training Time with different epoch = 3 and epoch = 5
Since default of sync_every == 0, so with MPI, all process will sync at end of each epoch.

- ReLU and Epoch = 3
```bash
mpiexec -host 192.168.1.7:10 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 625 --activation relu --epochs 3 : \
        -host XuanNguyen@192.168.1.9:6 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation relu --epochs 3
```
- ReLU and Epoch = 5
```bash
mpiexec -host 192.168.1.7:10 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 625 --activation relu --epochs 5 : \
        -host XuanNguyen@192.168.1.9:6 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation relu --epochs 5
```
- Tanh and Epoch = 3
```bash
mpiexec -host 192.168.1.7:10 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 625 --activation tanh --epochs 3 : \
        -host XuanNguyen@192.168.1.9:6 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation tanh --epochs 3
```
- Tanh and Epoch = 5
```bash
mpiexec -host 192.168.1.7:10 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 310 --activation tanh --epochs 5 : \
        -host XuanNguyen@192.168.1.9:6 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation tanh --epochs 5
```
- Sigmoid and Epoch = 3
```bash
mpiexec -host 192.168.1.7:10 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 625 --activation sigmoid --epochs 3 : \
        -host XuanNguyen@192.168.1.9:6 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation sigmoid --epochs 3
```
- Sigmoid and Epoch = 5
```bash
mpiexec -host 192.168.1.7:10 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 625 --activation sigmoid --epochs 5 : \
        -host XuanNguyen@192.168.1.9:6 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation sigmoid --epochs 5

```
Trial 2:  Experience to setup the MPI to sync at every 1000 batch (within epoch), instead of only sync at one time after each process finish at the end of an epoch

```bash
mpiexec -host 192.168.1.7:5 venv/bin/python 03_MPI_SGD_NN_train_v1.py --data data/output/split_data --batch-size 1024 --print-every 310 --activation relu --epochs 1 --sync-every 1000 : \
        -host XuanNguyen@192.168.1.9:3 /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/venv/bin/python /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/03_MPI_SGD_NN_train_v1.py --data /Users/XuanNguyen/Documents/NUS/DSA5208/DSA5208_Project1/data/output/split_data --batch-size 1024 --print-every 100000 --activation relu --epochs 1 --sync-every 1000

```
## 9) run report (program: z_extract_log_to_csv.ipynb)

After all CLI to experiment different config combination for model. run command in program: z_extract_log_to_csv.ipynb to able to produce record history/result in csv and produce chart (result) from record csv, which are used to build report.