import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.load_digits(return_X_y=True)

boolspocflbn = True

phas, inptflbn, outp, legdoutp, tici, itoi = exopmain.retr_datatess(boolspocflbn) 

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(phas, outp, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


