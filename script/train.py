#!/usr/bin/env python


from helper import ReadRecipe, NaiveBayes, RandomForest
from helper import AdaBoost, LogisticReg, SVC


train_ifn = "../data/train.json"

read_recipe = ReadRecipe(train_ifn)

train_raw, encoder = read_recipe.readTrain()


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
    pass


if __name__ == '__main__':
    main()
