import json
import re
import uuid
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, load_dataset
from tqdm import tqdm

dataset = load_dataset("Team-ACE/ToolACE")
new_data = {"id": [], "conversations": []}

# Process each example
for example in dataset["train"]:
    # Add system message to conversations and create new structure
    new_data["id"].append(str(uuid.uuid4()))
    new_data["conversations"].append(
        [{"from": "system", "value": example["system"]}] + example["conversations"]
    )
new_dataset = Dataset.from_dict(new_data)

# Save it
new_dataset.save_to_disk("transformed_toolace-new")
