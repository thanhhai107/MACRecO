#! /bin/bash

## Test on MovieLens-100k
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task sr --samples 100
python main.py --main Test --data_file data/ml-100k/test.csv --system collaboration --system_config config/systems/collaboration/reflect_analyse.json --task sr --samples 100

### Use --data_file to specify the data file example: data/amazon/Beauty/test.csv or data/amazon/Electronics/test.csv
### Use --main Evaluate and withou --samples to evaluate full dataset
### Use --system_config to specify the system configuration example: config/systems/collaboration/analyse.json or config/systems/collaboration/reflect_analyse.json
### Use --task to specify the task example: sr or rp

# Calculate the metrics directly from the run data file
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-vicu-0.jsonl
python main.py --main Calculate --task rp --run_data_file results/ml-100k/rp/gpt-vicu-1.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-gpt.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-vicu-0.jsonl
python main.py --main Calculate --task rp --run_data_file results/Beauty/rp/gpt-vicu-1.jsonl
