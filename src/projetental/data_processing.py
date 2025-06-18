import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Dict, Optional


def extract_instances_to_dataframe(file_path: str):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for text in root.findall("text"):
        for sentence in text.findall("sentence"):
            tokens = []
            target_info = None

            for element in sentence:
                if element.tag == "wf":
                    tokens.append(element.text)
                elif element.tag == "instance":
                    target_index = len(tokens)
                    tokens.append(element.text)
                    target_info = {
                        "lemma": element.attrib["lemma"],
                        "surface": element.text,
                        "pos": element.attrib["pos"],
                        "id": element.attrib["id"],
                        "target_index": target_index,
                    }

            if target_info:
                data.append(
                    {
                        "lemma": target_info["lemma"],
                        "sentence": " ".join(tokens),
                        "target_index": target_info["target_index"],
                        "target_form": target_info["surface"],
                        "target_id": target_info["id"],
                    }
                )

    return pd.DataFrame(data)


def load_gold_labels(key_file_path: str):
    id2sense = {}
    with open(key_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id2sense[parts[0]] = parts[1]
    return id2sense
