"""
tpp_data_converter.py
Converts our JSON sequences into EasyTPP's expected data format.
Each record needs: time_since_start, time_since_last_event, type_event, dim_process
"""

import json
import os

IN_PATH  = r"D:\Egna projekt\h01-sepsis\data\tpp"
OUT_PATH = r"D:\Egna projekt\h01-sepsis\data\tpp\easytpp"
NUM_EVENT_TYPES = 14


def convert(in_file: str, out_file: str):
    with open(in_file, 'r') as f:
        sequences = json.load(f)

    converted = []
    for seq in sequences:
        converted.append({
            'time_since_start':      seq['time_since_start'],
            'time_since_last_event': seq['time_since_last'],
            'type_event':            seq['event_type'],
            'dim_process':           NUM_EVENT_TYPES,
            'label':                 seq['label']
        })

    with open(out_file, 'w') as f:
        json.dump(converted, f)

    print(f"Converted {len(converted)} sequences → {out_file}")


def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    convert(
        os.path.join(IN_PATH, 'train_setA.json'),
        os.path.join(OUT_PATH, 'train.json')
    )
    convert(
        os.path.join(IN_PATH, 'test_setA.json'),
        os.path.join(OUT_PATH, 'dev.json')
    )
    convert(
        os.path.join(IN_PATH, 'test_setB.json'),
        os.path.join(OUT_PATH, 'test.json')
    )
    print("Conversion complete.")


if __name__ == '__main__':
    main()