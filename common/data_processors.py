import os
import json



class DataProcessor(object):
    def __init__(self, dataset_config):
        with open(dataset_config, "r", encoding='utf-8') as f:
            raw_config = json.load(f)