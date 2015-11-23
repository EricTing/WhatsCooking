import json
import pandas as pd
from math import log10
from itertools import chain


train_ifn = "../data/train.json"
test_ifn = "../data/test.json"


def readRawDset(ifn):
    def read(ifn):
        return json.loads(open(ifn, 'r').read())

    train_data = read(ifn)
    cuisine_and_ingreds = [(_[u'cuisine'], _["ingredients"])
                           for _ in train_data]
    return pd.DataFrame(cuisine_and_ingreds,
                        columns=['cuisine', 'ingredients'])

raw_dset = readRawDset(train_ifn)


# ## tf-idf method (term frequency–inverse document frequency)
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf

def tf(tokens):
    '''take a list of tokens and returns a Python dictionary mapping tokens to TF weights
    '''
    from collections import Counter
    return Counter(tokens)


tfs = {}

for name, group in raw_dset.groupby('cuisine'):
    ingredients = list(chain(*group.ingredients.values))
    tfs[name] = tf(ingredients)


def getTf(cuisine, ingredient):
    return tfs[cuisine][ingredient]

# ### calculate inverse document frequence
cuisines_ingredients = {}  # the ingredients of each cuisine
all_ingredients = list(chain(*raw_dset.ingredients.values))

for name, group in raw_dset.groupby('cuisine'):
    ingredients = list(chain(*group.ingredients.values))
    cuisines_ingredients[name] = set(ingredients)

idf = {}


for ingredient in set(all_ingredients):
    num_cuisines_contains_this_ingredient = len([_ for _ in cuisines_ingredients
                                                 if ingredient in cuisines_ingredients[_]])
    idf[ingredient] = log10(len(set(raw_dset.cuisine)) / num_cuisines_contains_this_ingredient)


cuisines = set(raw_dset.cuisine)


def calcualteIfIdfOfIngredient(ingredient, cuisine):
    return idf[ingredient] * tfs[cuisine][ingredient]


def calculateTfIdfOfCuisine(ingredients, cuisine):
    return sum((calcualteIfIdfOfIngredient(ingredient, cuisine)
                for ingredient in ingredients))


def rankByTfIdf(ingredients):
    freqs = []
    for cuisine in cuisines:
        freqs.append((cuisine, calculateTfIdfOfCuisine(ingredients, cuisine)))
    freqs.sort(key=lambda t: t[1], reverse=True)
    return freqs[0][0]


## ranking based on tf-idf value
predicted_cuisines = raw_dset.ingredients.apply(lambda ingredients: rankByTfIdf(ingredients))
accuracy = sum(predicted_cuisines == raw_dset.cuisine) / float(len(raw_dset.cuisine))
print accuracy
