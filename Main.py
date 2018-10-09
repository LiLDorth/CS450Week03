import sklearn

import numpy
import scipy
from numpy.distutils.fcompiler import none
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

class Model:


    def trained(self):
        #print(*self.Train_Data, sep="\n")
        for item in self.Train_Data:
            print(item[0][0])

    def predict(self, data):
        prediction = []
        distances = []
        for item in data:
            distances.clear()
            for trainedItem in self.Train_Data:
                distances.append([(abs((item[0] - trainedItem[0][3])) + abs((item[1] - trainedItem[0][1])) + abs((item[2] - trainedItem[0][2])) + abs((item[3] - trainedItem[0][3]))), trainedItem[1]])
            distances.sort()
            targetNeighbors = []
            for closest in distances[:self.K]:
                targetNeighbors.append(closest[1])
            prediction.append(Counter(targetNeighbors).most_common()[0][0])
        return prediction


class HardcodedClassifier:
    def fit(X_Train, Y_Train, k):
        #print(X_Train, Y_Train)
        Model.Train_Data = list(zip(X_Train, Y_Train))
        Model.K = k
        return Model

def main():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    encoder = preprocessing.LabelEncoder()
    #AUTISM DATASET
    autism_headers = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age', 'Gender', 'Ethnicity', 'Jundice', 'Autism', 'Country', 'Used app before', 'Result', 'Age Description', 'Relation', 'Class/ASD']
    autism = pd.io.parsers.read_csv('Autism-Adult-Data.csv', names=autism_headers, na_values="?")
    autism.drop('Age Description', axis=1, inplace=True)
    autism.replace(['no', "NO","m", "YES", 'yes', "f"], [0, 0, 0, 1, 1, 1], inplace=True)
    autism['Ethnicity'] = encoder.fit_transform(autism['Ethnicity'].astype(str))
    autism['Country'] = encoder.fit_transform(autism['Country'].astype(str))
    autism['Relation'] = encoder.fit_transform(autism['Relation'].astype(str))
    #print(autism.dtypes)
    autism = autism.dropna()
    target = autism['Class/ASD']
    autism.drop('Class/ASD', axis=1, inplace=True)

    scaler = StandardScaler()
    autism = scaler.fit_transform(autism)
    data_train, data_test, target_train, target_test  = train_test_split(autism, target, test_size=.30)
    classifier = KNeighborsClassifier(n_neighbors=7)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)
    print("Autism No-KFold Accuracy: ", str(accuracy_score(target_test, targets_predicted)*100)+"%")

    k_fold = KFold(len(target), shuffle=True, random_state=7)
    y_pred = cross_val_predict(classifier, autism, target, cv=k_fold, n_jobs=1)
    autism_accuracy_score = cross_val_score(classifier, autism, target, cv=k_fold).mean()

    print("Autism K-Fold Accuracy: ", autism_accuracy_score * 100, "%")

#MPG DATASET
    mpg_headers = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']
    mpg_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    mpg = pd.read_csv(mpg_data, names=mpg_headers, na_values="?", delim_whitespace=True)
    mpg = mpg.dropna()
    mpg['Car Name'] = encoder.fit_transform(mpg['Car Name'].astype(str))
    mpg = mpg.astype(int)
    mpg['MPG'].replace(range(1,15), 0, inplace=True)
    mpg['MPG'].replace(range(15, 20), 1, inplace=True)
    mpg['MPG'].replace(range(20, 25), 2, inplace=True)
    mpg['MPG'].replace(range(25, 30), 3, inplace=True)
    mpg['MPG'].replace(range(30, 35), 4, inplace=True)
    mpg['MPG'].replace(range(35, 40), 5, inplace=True)
    mpg['MPG'].replace(range(40, 45), 6, inplace=True)
    mpg['MPG'].replace(range(45, 60), 7, inplace=True)
    mpg_target = mpg['MPG']
    mpg.drop('MPG', axis=1, inplace=True)

    scaler = StandardScaler()
    mpg = scaler.fit_transform(mpg)

    data_train, data_test, target_train, target_test = train_test_split(mpg, mpg_target, test_size=.30)

    classifier = KNeighborsClassifier(n_neighbors=13)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)

    print("MPG No-KFold Accuracy: ", str(accuracy_score(target_test, targets_predicted)*100)+"%")
    mpg_k_fold = KFold(len(mpg_target), shuffle=True, random_state=41)
    mpg_accuracy_score = cross_val_score(classifier, mpg, mpg_target, cv=mpg_k_fold).mean()
    print("MPG K-Fold Accuracy: ", mpg_accuracy_score * 100, "%")

#Cars reading and standardizing Data using Pandas built in dummy encoder
    cars_headers = ['Buying', 'Maintence', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'Target']
    car_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    cars = pd.read_csv(car_data, names=cars_headers, na_values="?")
    cars_target = cars['Target']
    cars.drop('Target', axis=1, inplace=True)
    cars = pd.get_dummies(cars).astype(int)

    scaler = StandardScaler()
    cars = scaler.fit_transform(cars)
    data_train, data_test, target_train, target_test = train_test_split(cars, cars_target, test_size=.30)
    classifier = KNeighborsClassifier(n_neighbors=17)
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)
    print("Cars No-KFold Accuracy: ", str(accuracy_score(target_test, targets_predicted)*100)+"%")

    cars_k_fold = KFold(len(mpg_target), shuffle=True, random_state=7)
    cars_y_pred = cross_val_predict(classifier, cars, cars_target, cv=cars_k_fold, n_jobs=1)
    cars_accuracy_score = cross_val_score(classifier, cars, cars_target, cv=cars_k_fold).mean()
    print("Cars K-Fold Accuracy: ", cars_accuracy_score * 100, "%")

#VISUALS

    #classes = datasets.load_iris().target_names
    #visualizer = ClassificationReport(classifier, classes=classes)
    #visualizer.fit(data_train, target_train)
    #visualizer.score(data_test, target_test)
    #visualizer.poof()

    #cm = ConfusionMatrix(classifier)
    #cm.fit(data_train, target_train)
    #cm.score(data_test, target_test)
    #cm.poof()
if __name__ == '__main__':
    main()

