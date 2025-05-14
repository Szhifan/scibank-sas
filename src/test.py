from data_utils import SB_Dataset
import json
import os
sb = SB_Dataset()


output_path = "data/merged_responses.json"
new_data = {}

# Iterate through all files in the directory
directory = "data"
for filename in os.listdir(directory):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, item in enumerate(lines):
                item = json.loads(item)
                id = list(item.keys())[0]
                new_data[id] = item[id]

# Write all merged data to the output file
with open(output_path, "w") as f:
    json.dump(new_data, f, indent=4)