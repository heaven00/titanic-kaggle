from data_model import Data
import pandas as pd
from sklearn.svm import SVC
import numpy as np


class TitanicData(Data):
    """Extend the Data class by customizing a preprocessing function"""

    def _preprocess(self):
        """Preprocess the titanic trainset"""
        data = self.data
        data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1, inplace=True)
        replacer = {'S': 10, 'C': 11, 'Q': 12}
        data.Embarked = data.Embarked.apply(lambda x: replacer[x] if pd.notnull(x) else np.nan)
        data.Sex = data.Sex.apply(lambda x: 0 if x == 'male' else 1)
        data.dropna(inplace=True)
        self.data = data
        

if __name__ == "__main__":
    dataset = TitanicData('data/train.csv')
    trainset = dataset.train()
    y = trainset['Survived']
    X = trainset.drop('Survived', axis=1)

    clf = SVC(kernel='rbf')
    clf.fit(X, y)
    validationset = dataset.validation()
    X_v = validationset.drop('Survived', axis=1)
    y_v = validationset['Survived']
    print clf.score(X_v, y_v)
    
    
