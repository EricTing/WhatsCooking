from re import sub
from pandas import Series
from pandas import DataFrame

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def clean(ingredient):
    # first letter should be alphabet and remaining letter may be alpha numerical
    return map(lambda x: sub('[^A-Za-z0-9]+', ' ', x), ingredient)


class ReadRecipe:
    def __init__(self, json_ifn):
        import json
        json_content = json.loads(open(json_ifn).read())
        self.json = json_content

    def readTrain(self):
        cuisines = [my['cuisine'] for my in self.json]
        ingredients = [my['ingredients'] for my in self.json]
        ingredients = [', '.join(clean(i)) for i in ingredients]

        dset = DataFrame({"cuisine": cuisines,
                          "ingredients": ingredients})

        encoder = LabelEncoder()
        dset['cuisine'] = encoder.fit_transform(dset['cuisine'])

        return dset, encoder


def ingredientModel(train_raw, cv):
    ingred_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(strip_accents='unicode', analyzer="char")),
        ('feat', SelectPercentile(chi2, percentile=85)),
        ('model', LogisticRegression())
    ])

    train_idx, test_idx = list(cv)[0]

    my_train = train_raw.iloc[train_idx]
    tokens = my_train.ingredients.str.split(',? ').apply(lambda word_list: Series(word_list)).stack().reset_index(level=1, drop=True)
    tokens.name = 'ingredient'
    my_train = my_train[['cuisine']].join(tokens)

    ingred_pipe.fit(my_train['ingredient'], my_train['cuisine'])

    def probaOfEachCuisin(recipe, model):
        from operator import add
        probas = [model.predict_proba([i]) for i in recipe]
        proba_of_each = reduce(add, probas)[0]
        return proba_of_each.argmax(0)

    my_test = train_raw.iloc[test_idx]
    predicted = my_test['ingredients'].str.split(',? ').apply(lambda recipe: probaOfEachCuisin(recipe, ingred_pipe))
    score = accuracy_score(my_test.cuisine, predicted)
    print score
