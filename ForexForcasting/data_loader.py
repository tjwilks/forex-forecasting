import os
import re
import pandas as pd


class ForexLoader:

    def load(self, source_type, path):
        if source_type == 'dir':
            data = self.load_from_dir(path)
        elif source_type == 'csv':
            data = self.load_from_csv(path)
        else:
            raise ValueError("ForexLoader load method's source_type argument"
                             'must be either "dir" or "csv"')
        return data

    def load_from_dir(self, path):
        filenames = [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        datasets = []
        for file in filenames:
            data = self.load_from_csv(file)
            datasets.append(data)
        return pd.concat(datasets, axis=0)

    def load_from_csv(self, path):
        data = pd.read_csv(path)
        data['currency_pair'] = f"USD/{os.path.basename(path)[:3]}"
        return data
