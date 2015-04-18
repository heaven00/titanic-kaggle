import pandas as pd
from collections import namedtuple


class Data(object):
    """A class to manipulate data using pandas for ML"""
    
    def __init__(self, filepath):
        """Initialize the class with file path to the dataset"""
        self.filepath = filepath
        self.data = self.read_data(filepath)
        self._preprocess()
        self.dataset = self.makedataset()

    def read_data(self, filepath):
        """read the data file."""
        if filepath.split('.')[-1] == 'csv':
            return pd.read_csv(filepath, low_memory=False)
        else:
            raise Exception('Please change the function according to file extension')        

    def _preprocess(self):
        """preprocess the data"""
        # add the preprocessing functionality here
        pass

    def train(self):
        """Return a trainset from the data"""
        return self.dataset.train
        

    def validation(self):
        """Return validation set from the data."""
        return self.dataset.validation

    def makedataset(self):
        """Slpit the data into validation and train set and return a tuple"""
        dataset = namedtuple('dataset', ['train', 'validation'])
        data = self.data
        eighty = int(len(data.index) * 0.8)
        train = data[:eighty]
        validation = data[eighty:]
        return dataset._make([train, validation])
        
        
if __name__ == "__main__":
    d = Data('data/train.csv')
    print d.train().head()

    
