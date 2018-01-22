from sklearn.model_selection import cross_val_score


def accuracy(preds, labels):
    total = 0.0
    correct = 0.0
    for i in range(preds.shape[0]):
        if preds[i] == labels[i]: correct += 1
        total += 1
    print(correct, total, round(correct / total, 3))


def mean(numbers):
    return round(float(sum(numbers)) / max(len(numbers), 1), 3)


def cross_valid(clf, train_X, train_Y, cv=5):
    scores = cross_val_score(clf, train_X, train_Y, cv=cv)
    print(mean(scores), scores)


def parameter_search(clf, train_X, train_Y, parameters=[], cv=5):
    for i in parameters:
        clf.n_estimators = i
        clf.fit(train_X, train_Y)
        print("hyper parameter:", i)
        preds0 = clf.predict(train_X)
        accuracy(preds0, train_Y)
        cross_valid(clf, train_X, train_Y, cv=cv)
