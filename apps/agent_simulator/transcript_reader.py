import argparse
import json
import os
import copy
import random
from datetime import datetime
import pandas as pd

def parse_opts():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="""Generate 
                                     an experiment config and the data needed to run main.py.""")
    
    parser.add_argument("--input-dir", 
                        help="the template for the experiment config", 
                        required=True)
    parser.add_argument("--output-file", 
                        help="the output file with the results", 
                        required=True)

    # Parse arguments
    args = parser.parse_args()
    return args


def load_json_file(filename):
    """
    Load and return the data from a JSON file.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError:
        raise RuntimeError(f"The file {filename} is not a valid JSON file.")

def find_json_files(root_dir, prefix):
    """ Recursively find all json files with a 
    given prefix in root_dir. """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith(prefix) and file.endswith('.json'):
                yield os.path.join(root, file)

def extract_data_from_json(file_path):
    """ Extracts specific data from a json file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
        prod_desc = data["config"]["agent_gpt_config"]["prompt_preamble"].split("##")[4].split("\n")[1:4]
        prod_desc = "\n".join(prod_desc)
        style = data["config"]["human_gpt_config"]["prompt_preamble"].split("## STYLE ##")[1].split("\n")[1]
        campaign_name = data["config"]["campaign_name"]

        return {
            "transcript": data['transcript'],
            "model_name": data["config"]["agent_gpt_config"]["gpt_model_name"],
            "style": style,
            "campaign_name": campaign_name,
            "prod_description": prod_desc
        }

def main():
    args = parse_opts()
    
    input_dir = args.input_dir
    output_file = args.output_file

    data = []
    for file_path in find_json_files(input_dir, 'transcript'):
        extracted_data = extract_data_from_json(file_path)
        data.append(extracted_data)
        print(extracted_data)

        input("Press Enter to continue to the next file...")

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
    