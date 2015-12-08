#!/usr/bin/env python


from helper import ReadRecipe, NaiveBayes, RandomForest
from helper import AdaBoost, LogisticReg, SVC
from helper import predict


train_ifn = "../data/train.json"

read_recipe = ReadRecipe(train_ifn)

train_raw, encoder = read_recipe.readTrain()

test_ifn = "../data/test.json"

test_recipe = ReadRecipe(test_ifn)

test_raw = test_recipe.readTest()


def main():
    # print "Naive Bayes"
    # NaiveBayes(train_raw, cv=6, parallism=20)
    # print "Random Forest"
    # RandomForest(train_raw, cv=6, parallism=20)
    # print "AdaBoostClassifier"
    # AdaBoost(train_raw, cv=6, parallism=20)
    # print "LogisticRegression"
    # LogisticReg(train_raw, cv=6, parallism=20)
    # print "linearSVC"
    # SVC(train_raw, cv=6, parallism=20)
    predict(train_raw, test_raw, encoder)
    pass


if __name__ == '__main__':
    main()
