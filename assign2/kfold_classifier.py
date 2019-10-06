import sys
import csv
from numpy import mean, array
from sklearn import tree, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
    print("Fetched data set")

    decision_tree_classifier = tree.DecisionTreeClassifier(min_samples_split=250)
    svm_classifier = svm.SVC()
    naive_bayes_classifier = MultinomialNB()

    print("Evaluating classifiers")
    
    decision_tree_scores = cross_val_score(decision_tree_classifier, data, target, cv=10)
    print("Decision tree mean score : ", mean(array(decision_tree_scores)))

    naive_bayes_scores = cross_val_score(naive_bayes_classifier, data, target, cv=10)
    print("Naive Bayes mean score : ", mean(array(naive_bayes_scores)))

    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data)
    # svm_scores = cross_val_score(svm_classifier, scaled_data, target, cv=10)
    # print("SVM mean score : ", mean(array(svm_scores)))
    

if __name__ == "__main__":
    main()