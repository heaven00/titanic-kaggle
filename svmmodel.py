from data_model import Data
import pandas as pd
from sklearn.svm import SVC
import numpy as np


class TitanicData(Data):
    """Extend the Data class by customizing a preprocessing function"""

    
    def process_name(self):
        """Process the Name column and convert it into features"""
        names = self.data.Name
        names = names.apply(lambda x: x.split(',')[1].split()[0].replace('.', ''))
        keystoval = {'Mr': 1, 'Mrs': 2, 'Miss':3,
                     'Master': 4, 'Rev':5, 'Dr':6}
        names = names.apply(lambda x: keystoval[x] if x in keystoval.keys() else 50)
        leftovers = [index for index, val in enumerate(names.tolist()) if val not in [1,2,3,4]]
        print len(leftovers)
        return names
        
        
    def _preprocess(self):
        """Preprocess the titanic trainset"""
        data = self.data
        data['Name'] = self.process_name()
        data.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
        replacer = {'S': 10, 'C': 11, 'Q': 12}
        data.Embarked = data.Embarked.apply(lambda x: replacer[x] if pd.notnull(x) else np.nan)
        data.Sex = data.Sex.apply(lambda x: 0 if x == 'male' else 1)
        data = data.dropna()
        self.data = data


if __name__ == "__main__":
    dataset = TitanicData('data/train.csv')
    trainset = dataset.train()
    y = trainset['Survived']
    columns = ['Sex', 'Pclass', 'Name']
    print trainset['Name'].unique()
    X = trainset[columns]
    clf = SVC(kernel='rbf')
    clf.fit(X, y)
    validationset = dataset.validation()
    X_v = validationset[columns]
    y_v = validationset['Survived']
    print clf.score(X_v, y_v)
    # test the model on the test set and save submission.csv
    testset = Data('data/test.csv').data
    passengers = testset['PassengerId']
    testset.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)
    replacer = {'S': 10, 'C': 11, 'Q': 12}
    testset.Embarked = testset.Embarked.apply(lambda x: replacer[x] if pd.notnull(x) else np.nan)
    testset.Sex = testset.Sex.apply(lambda x: 0 if x == 'male' else 1)
    names = testset.Name
    names = names.apply(lambda x: x.split(',')[1].split()[0].replace('.', ''))
    keystoval = {'Mr': 1, 'Mrs': 2, 'Miss':3,
                 'Master': 4, 'Rev':5, 'Dr':6}
    names = names.apply(lambda x: keystoval[x]
                        if x in keystoval.keys() else 50)
    testset.Name = names
    testset = testset.fillna(0)
    testset = testset[columns]
    prediction = clf.predict(testset)
    pd.DataFrame({'Survived':prediction,
                  'PassengerId':passengers}).to_csv('submission.csv', index=False)



