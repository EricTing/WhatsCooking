import json
import pandas as pd
import numpy as np
from math import log10
from itertools import chain
from sklearn.metrics import pairwise_distances


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
    if cuisine in tfs and ingredient in tfs[cuisine]:
        return tfs[cuisine][ingredient]
    else:
        return 0.0

# ### calculate inverse document frequence
cuisines_ingredients = {}  # the ingredients of each cuisine
all_ingredients = list(chain(*raw_dset.ingredients.values))

for name, group in raw_dset.groupby('cuisine'):
    ingredients = list(chain(*group.ingredients.values))
    cuisines_ingredients[name] = set(ingredients)

idf = {}
total_cuisines = len(set(raw_dset.cuisine))
for ingredient in set(all_ingredients):
    num_cuisines_contains_this_ingredient = float(len([_ for _ in cuisines_ingredients
                                                       if ingredient in cuisines_ingredients[_]]))
    idf[ingredient] = log10(total_cuisines / num_cuisines_contains_this_ingredient)


def getIdf(ingredient):
    return idf[ingredient]


cuisines = set(raw_dset.cuisine)


def calcualteIfIdfOfIngredient(ingredient, cuisine):
    return getIdf(ingredient) * getTf(cuisine, ingredient)


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

## vector space model
## Scoring, term weighting, and the vector space model, Section 6.3.1
### ranking by the similarity between the query ingredients' vector and the cuisines' vector

available_ingredients = list(set(all_ingredients))
indices = {ingredient: available_ingredients.index(ingredient)
           for ingredient in available_ingredients}

def calculateCuisinVectors():
    """entries are the tf-idf values
    """
    cuisine_vectors = {}
    for cuisine in set(raw_dset.cuisine):
        my_vector = [0.0] * len(available_ingredients)
        for ingredient in available_ingredients:
            my_vector[indices[ingredient]] = calcualteIfIdfOfIngredient(ingredient, cuisine)

        cuisine_vectors[cuisine] = my_vector

    return cuisine_vectors

cuisine_vectors = calculateCuisinVectors()

def calculateVectorIfBelongs2ThisCuisine(cuisine, ingredients):
    my_vector = [0.0] * len(available_ingredients)
    for ingredient in available_ingredients:
        my_vector[indices[ingredient]] = calcualteIfIdfOfIngredient(ingredient, cuisine)

    return my_vector


def calculateIngredientsVector(ingredients):
    my_vector = [0.0] * len(available_ingredients)
    for ingredient in ingredients:
        my_vector[indices[ingredient]] = 1
    return my_vector


def calculateSim(v1, v2):       # cannot give correct measure of similarity
    return pairwise_distances(v1, v2, metric='cosine')


def rankByVecSim(ingredients):
    sims = []
    for cuisine in cuisines:
        cuisine_vector = cuisine_vectors[cuisine]
        ingredients_vector = calculateIngredientsVector(ingredients)
        sim = np.dot(cuisine_vector, ingredients_vector) /(np.linalg.norm(cuisine_vector) * np.linalg.norm(ingredients_vector))
        sims.append((cuisine, sim))
        # print cuisine, sim
    sims.sort(key=lambda t: t[1], reverse=True)
    return sims[0][0]

test_dset = raw_dset.sample(5000)
predicted_cuisines = test_dset.ingredients.apply(lambda ingredients: rankByVecSim(ingredients))
accuracy = sum(predicted_cuisines == test_dset.cuisine) / float(len(test_dset.cuisine))
print accuracy

