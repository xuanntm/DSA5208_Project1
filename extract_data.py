import pandas as pd

# Define the input and output file paths
input_file = 'sample_data/nytaxi2022.csv'
output_file = 'sample_data/nytaxi2022_focus.csv'

try:
    # Read the first 1000 rows from the CSV file
    focus_columns = ['tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count','trip_distance','RatecodeID','PULocationID','DOLocationID','payment_type','extra','total_amount']
    
    # df = pd.read_csv(input_file, nrows=5000000)
    df = pd.read_csv(input_file, usecols=focus_columns)
    df.to_csv(output_file, index=False)

    # Select only these columns from the DataFrame
    # df_filtered = df[focus_columns]

    # Save the resulting DataFrame to a new CSV file
    # df.to_csv(output_file, index=False)
    # df_filtered.to_csv(output_file, index=False)
    
    print(f'Successfully extracted the first 1000 records from "{input_file}"')
    print(f'Saved the records to "{output_file}"')

except FileNotFoundError:
    print(f'Error: The file "{input_file}" was not found.')
except Exception as e:
    print(f'An error occurred: {e}')