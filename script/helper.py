from re import sub
from pandas import Series
from pandas import DataFrame

from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


def porter(text):
    tokens = word_tokenize(text)

    my_stemmer = PorterStemmer()
    tokens = [my_stemmer.stem(t) for t in tokens]

    return tokens


def wordnet(text):
    tokens = word_tokenize(text)

    my_stemmer = WordNetLemmatizer()
    tokens = [my_stemmer.lemmatize(t) for t in tokens]

    return tokens


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
        ('feat', SelectPercentile(chi2)),
        ('model', LogisticRegression())
    ])

    ingred_grid = {
        'tfidf__ngram_range': [(2, 4)],
        'feat__percentile': [95, 90, 85]
    }

    def score4OneMetaParameterSet(train_raw, cv, grid_paras):
        ingred_pipe.set_params(**grid_paras)

        total_score = 0.0
        for train_idx, test_idx in cv:
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
            total_score += score

        return total_score

    best_score = 0
    for g in ParameterGrid(ingred_grid):
        score = score4OneMetaParameterSet(train_raw, cv, g)
        if score > best_score:
            best_score = score
            best_grid = g

    print "Best score: %0.5f" % best_score
    print "Best grid:", best_grid


def NaiveBayes(train_raw, cv=6, parallism=20):
    pipe_line = Pipeline([
        ('tfidf', TfidfVectorizer(strip_accents='unicode', analyzer="char")),
        ('feat', SelectPercentile(chi2)),
        ('model', MultinomialNB())
    ])

    grids = {
        'tfidf__ngram_range': [(2, 4), (2, 5), (2, 6)],
        'feat__percentile': [95, 90, 85, 80, 75, 70]
    }

    grid_search = GridSearchCV(pipe_line, grids, n_jobs=parallism,
                               verbose=1, cv=cv)
    grid_search.fit(train_raw.ingredients, train_raw.cuisine)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(grids.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def RandomForest(train_raw, cv=6, parallism=20):
    pipe_line = Pipeline([
        ('tfidf', TfidfVectorizer(
            strip_accents='unicode',
            stop_words='english')),
        ('feat', SelectPercentile(chi2)),
        ('model', RandomForestClassifier(n_estimators=50))
    ])

    grids = {
        # 'tfidf__tokenizer': [porter, wordnet],
        'tfidf__tokenizer': [wordnet],

        'tfidf__max_df': [0.5, 0.4, 0.3, 0.2],

        # 'tfidf__analyzer': ["char", "word"],
        'tfidf__analyzer': ["word"],

        # 'tfidf__ngram_range': [(1, 1), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)],
        'tfidf__ngram_range': [(1, 1)],

        'feat__percentile': [50, 40, 30, 20, 10]
    }

    grid_search = GridSearchCV(pipe_line, grids, n_jobs=parallism,
                               verbose=2, cv=cv)
    grid_search.fit(train_raw.ingredients, train_raw.cuisine)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(grids.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def AdaBoost(train_raw, cv=6, parallism=20):
    pipe_line = Pipeline([
        ('tfidf', TfidfVectorizer(
            strip_accents='unicode',
            stop_words='english')),
        ('feat', SelectPercentile(chi2)),
        ('model', AdaBoostClassifier(
            algorithm="SAMME"))
    ])

    grids = {
        'tfidf__tokenizer': [wordnet],

        'tfidf__max_df': [0.5, 0.4, 0.3, 0.2, 0.1],

        'tfidf__analyzer': ["word"],

        'tfidf__ngram_range': [(1, 1)],

        'feat__percentile': [50, 40, 30, 20, 10],

        'model__learning_rate': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],

        'model__n_estimators': [50, 100, 200]
    }

    grid_search = GridSearchCV(pipe_line, grids, n_jobs=parallism,
                               verbose=2, cv=cv)
    grid_search.fit(train_raw.ingredients, train_raw.cuisine)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(grids.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def LogisticReg(train_raw, cv=6, parallism=20):
    pipe_line = Pipeline([
        ('tfidf', TfidfVectorizer(
            strip_accents='unicode',
            stop_words='english')),
        ('feat', SelectPercentile(chi2)),
        ('model', LogisticRegression())
    ])

    grids = {
        'tfidf__tokenizer': [wordnet],

        'tfidf__max_df': [0.5, 0.4, 0.3, 0.2, 0.1],

        'tfidf__analyzer': ["word"],

        'tfidf__ngram_range': [(1, 1)],

        'feat__percentile': [50, 40, 30, 20, 10],

        'model__C': [1, 5, 10]
    }

    grid_search = GridSearchCV(pipe_line, grids, n_jobs=parallism,
                               verbose=2, cv=cv)
    grid_search.fit(train_raw.ingredients, train_raw.cuisine)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(grids.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
