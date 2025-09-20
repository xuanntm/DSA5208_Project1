import argparse
import csv
import sys
import os
from datetime import datetime

def log_time(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def split_csv_by_modulo(input_file, num_files, output_folder, output_prefix="part"):
    """
    Split a large CSV into multiple smaller files using row_index % num_files.
    - input_file: path to the big CSV file
    - num_files: number of output files
    - output_folder: path for output files
    - output_prefix: prefix for output files (default 'part')
    """
    # Open all output files at once (for performance)
    os.makedirs(f"{output_folder}/to_{num_files}", exist_ok=True)
    out_files = [open(f"{output_folder}/to_{num_files}/{output_prefix}_{i}.csv", "w", newline='', encoding="utf-8") for i in range(num_files)]
    writers = [csv.writer(f) for f in out_files]

    with open(input_file, "r", newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        
        # Read header
        header = next(reader, None)
        if header:
            for w in writers:
                w.writerow(header)

        # Distribute rows based on index % num_files
        for idx, row in enumerate(reader):
            writers[idx % num_files].writerow(row)

    # Close all output files
    for f in out_files:
        f.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input-file-path', type=str, required=True, help='Path to target file')
    p.add_argument('--output-folder', type=str, required=True, help='Path to target file')
    p.add_argument('--number-process', type=int, required=True, help='number of process')
    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    log_time(args)

    input_file = args.input_file_path
    num_files = args.number_process
    output_folder = args.output_folder

    split_csv_by_modulo(input_file, num_files, output_folder)
    log_time('Split completed')