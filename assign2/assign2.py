import sys
import csv
import numpy
from sklearn import tree, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


NUM_ATTRIBUTES = 42
ATTRIBUTE_VALUE_MAP = {
    'x' :0,
    'o' :1,
    'b' :2
}

CLASS_VALUE_MAP = {
    'win' : 0,
    'loss' : 1,
    'draw' : 2
}

def readData():
    attribute_data = []
    class_data = []
    with open("./data/connect-4.data") as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            if len(row) == NUM_ATTRIBUTES+1:
                attribute_data.append(row[:NUM_ATTRIBUTES])
                class_data.append(row[NUM_ATTRIBUTES])

    for i in range(len(attribute_data)):
        for j in range(NUM_ATTRIBUTES):
            attribute_data[i][j] = ATTRIBUTE_VALUE_MAP[attribute_data[i][j]]
    
    for i in range(len(class_data)):
        class_data[i] = CLASS_VALUE_MAP[class_data[i]]
    
    return attribute_data, class_data

def main():
    data, target = readData()

    decision_tree_classifier = tree.DecisionTreeClassifier(min_samples_split=250)
    svm_classifier = svm.LinearSVC()
    naive_bayes_classifier = MultinomialNB()

    decision_tree_scores = cross_val_score(decision_tree_classifier, data, target, cv=10)
    print(numpy.mean(numpy.array(decision_tree_scores)))
    

if __name__ == "__main__":
    main()