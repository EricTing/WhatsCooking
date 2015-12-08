# WhatsCooking

Use recipe ingredients to categorize the cuisine

https://www.kaggle.com/c/whats-cooking

## Naive Bayes
Best score: 0.667

Best parameters set:

feat__percentile: 70

tfidf__ngram_range: (2, 4)


## Random Forest
Best score: 0.752

Best parameters set:

feat__percentile: 40

tfidf__analyzer: 'word'

tfidf__max_df: 0.3

tfidf__ngram_range: (1, 1)

tfidf__tokenizer: <function wordnet at 0x2aecf6e88f50>

## AdaBoost
Why AdaBoost performs so poorly?

Best score: 0.549

Best parameters set:

feat__percentile: 10

model__learning_rate: 2.0

model__n_estimators: 200

tfidf__analyzer: 'word'

tfidf__max_df: 0.1

tfidf__ngram_range: (1, 1)

tfidf__tokenizer: <function wordnet at 0x2ae685d86050>


## LogisticRegression
Best score: 0.791

Best parameters set:

feat__percentile: 80

model__C: 5

tfidf__analyzer: 'word'

tfidf__max_df: 0.9

tfidf__ngram_range: (1, 1)

tfidf__tokenizer: <function wordnet at 0x2b065874f0c8>

