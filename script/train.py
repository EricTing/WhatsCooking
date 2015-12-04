#!/usr/bin/env python

from helper import ReadRecipe, ingredientModel
from sklearn.cross_validation import KFold


train_ifn = "../data/train.json"

read_recipe = ReadRecipe(train_ifn)

train_raw, encoder = read_recipe.readTrain()

cv = KFold(train_raw.shape[0], n_folds=6, shuffle=True)


def main():
    ingredientModel(train_raw, cv)
    pass

if __name__ == '__main__':
    main()
