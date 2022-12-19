import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree, svm
file = open("data.txt", "r")
data = []
for l in file.readlines():
    if "$$" in l:
        continue
    content = l.split(" ")
    sentence = []
    c = 0
    interest = 0
    for t in content:
        if "=" in t:
            continue
        if t == "[" or t == "]":
            continue
        if t == "\n":
            continue

        tt = t.split("/")
        if len(tt) == 2:
            # On trouve ou se trouve le mot "interest"
            if "interest" in t and "_" in t:
                interest = c
            sentence.append(tt)
            c += 1

    d = ["@" for _ in range(9)]
    d[0] = sentence[interest][0].split("_")[1]

    # On sauvegarde les 2 mots avant et après
    if (interest-1) >= 0:
        d[1] = sentence[interest-1][0]
        d[2] = sentence[interest-1][1]
    if (interest-2) >= 0:
        d[3] = sentence[interest-2][0]
        d[4] = sentence[interest-2][1]
    if (interest+1) < len(sentence):
        d[5] = sentence[interest+1][0]
        d[6] = sentence[interest+1][1]
    if (interest+2) < len(sentence):
        d[7] = sentence[interest+2][0]
        d[8] = sentence[interest+2][1]
    data.append(d)

t = []
a = []
for i in data:
    t.append(" ".join(i[1:]))
    a.append(i[0])


def custom(text):
    return text.split(" ")


vectorizer = CountVectorizer(tokenizer=custom)
X = vectorizer.fit_transform(t).toarray()
tokenizer = vectorizer.build_tokenizer()
output_corpus = []
na = []
for i, element in enumerate(t):
    element = tokenizer(element.lower())
    # On s'assure d'avoir le bon format.
    if(len(element) == 8):
        output_line = []
        for token in element:
            output_line.append(vectorizer.vocabulary_.get(token))
        output_corpus.append(output_line)
        na.append(a[i])


# Test de performance

# Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(
    output_corpus, na, test_size=0.5, random_state=0)
gnb = MultinomialNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Dumb Bayes Number of mislabeled points out of a total %d points : %d" %
      (len(X_test), (y_test != y_pred).sum()))

# Arbre seul
clf3 = tree.DecisionTreeClassifier()
clf3 = clf3.fit(output_corpus, na)
y_pred = clf3.predict(X_test)
print("TREE Number of mislabeled points out of a total %d points : %d" %
      (len(X_test), (y_test != y_pred).sum()))

# Forêt aléatoire
clf = RandomForestClassifier(n_estimators=150)
clf = clf.fit(output_corpus, na)
y_pred = clf.predict(X_test)
print("Forest Number of mislabeled points out of a total %d points : %d" %
      (len(X_test), (y_test != y_pred).sum()))

# SVM
clf4 = svm.SVC()
clf4 = clf4.fit(output_corpus, na)
y_pred = clf4.predict(X_test)
print("SVM Number of mislabeled points out of a total %d points : %d" %
      (len(X_test), (y_test != y_pred).sum()))

# MLP
clf2 = MLPClassifier(hidden_layer_sizes=(64, 64), random_state=0)
clf2 = clf2.fit(output_corpus, na)
y_pred = clf2.predict(X_test)
print("MLP Number of mislabeled points out of a total %d points : %d" %
      (len(X_test), (y_test != y_pred).sum()))
