from data_model import Data
import pandas as pd


class TitanicData(Data):
    """Extend the Data class by customizing a preprocessing function"""

    def _preprocess(self):
        """Preprocess the titanic trainset"""
        data = self.data
        data.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.data = data
        

if __name__ == "__main__":
    d = TitanicData('data/train.csv')
    processed_data = d.train()
    print processed_data.head()

