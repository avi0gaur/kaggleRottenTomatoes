from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
import csv
import numpy as np
import sys

X_train = []
y_train = []

ID_test = []
X_test = []

# Get training data
with open('../data/train.tsv', 'r') as f:
	f.readline()
	csvreader = csv.reader(f, delimiter='\t')
	for row in csvreader:
		X_train.append(row[2])
		y_train.append(row[3])

# Get test data:
with open('../data/test.tsv', 'r') as f:
	f.readline()
	csvreader = csv.reader(f, delimiter='\t')
	for row in csvreader:
		ID_test.append(row[0])
		X_test.append(row[2])

# Make transformation pipeline
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', BernoulliNB()),
                     #('clf', SGDClassifier())
])

# Set the parameters to be optimized in the Grid Search
parameters = {'vect__ngram_range': [(1, 1),(1,2)],
              'vect__stop_words': ('english', None),
              'vect__min_df': (1,2,3),
              # 'vect__max_features': (1000, 5000, 10000, 15000),
              'tfidf__use_idf': (True, False),
              # 'clf__alpha': (1, 0),
}

# Fit the grid search using all CPUs
gs_clf = GridSearchCV(text_clf, parameters, cv=10, n_jobs=-1)

# Transform and fit the model
gs_clf.fit(X_train, y_train)

# Transform the test data and get predictions
predictions = gs_clf.predict(X_test)

# Get the best parameters and print them so we know
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
print score

# Sanity check: make sure that the predictions array is the correct length
if len(ID_test) != len(predictions):
	raise StandardError("Test data error")
	sys.exit()

# Create the submission file
with open('../out/s9.csv', 'w') as outfile:
	outfile.write("PhraseId,Sentiment\n")
	for phrase_id,pred in zip(ID_test,predictions):
		outfile.write('{},{}\n'.format(phrase_id,pred))
