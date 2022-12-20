# TP2- IFT3335

# WILLIAM D'ANJOU 20188213
# VANESSA THIBAULT-SOUCY 20126808

import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree, svm
import matplotlib.pyplot as plt

# Lecture du data et traitement de chaque anotations.
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
    # Crée la structure de base pour une liste pour chaque anotations traités
    d = ["@" for _ in range(9)]
    d[0] = sentence[interest][0].split("_")[1]

    # On sauvegarde les 2 mots avant et après
    if (interest - 1) >= 0:
        d[1] = sentence[interest - 1][0]
        d[2] = sentence[interest - 1][1]
    if (interest - 2) >= 0:
        d[3] = sentence[interest - 2][0]
        d[4] = sentence[interest - 2][1]
    if (interest + 1) < len(sentence):
        d[5] = sentence[interest + 1][0]
        d[6] = sentence[interest + 1][1]
    if (interest + 2) < len(sentence):
        d[7] = sentence[interest + 2][0]
        d[8] = sentence[interest + 2][1]
    data.append(d)

t = []
a = []
for i in data:
    t.append(" ".join(i[1:]))
    a.append(""+i[0])


def graph(classifier, yTrain, yTest):
    t = sklearn.metrics.accuracy_score(
        yTrain, yTest)
    print(classifier + ": " + str(np.round(t, 5)))

    # plt.figure()
    # plt.show(figNB)


def custom(text):
    return text.split(" ")


# Traitement des listes en vecteurs
# inspiration: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#
vectorizer = CountVectorizer(tokenizer=custom)
X = vectorizer.fit_transform(t).toarray()
tokenizer = vectorizer.build_tokenizer()
output_corpus = []
na = []
for i, element in enumerate(t):
    element = tokenizer(element.lower())
    # On s'assure d'avoir le bon format.
    if (len(element) == 8):
        output_line = []
        for token in element:
            output_line.append(vectorizer.vocabulary_.get(token))
        output_corpus.append(output_line)
        na.append(a[i])

# Entrainement et test de performance des models

# Separation du jeu de données en 2 (train, test)
X_train, X_test, y_train, y_test = train_test_split(
    output_corpus, na, test_size=0.5, random_state=0)

# Naive Bayes
if False:
    dd = []
    x = [0.2, 0.5, 0.8]
    for y in x:
        avg = []
        for _ in range(3):
            X_train, X_test, y_train, y_test = train_test_split(
                output_corpus, na, test_size=y, random_state=0)
            bayesNaif = MultinomialNB()
            y_pred = bayesNaif.fit(X_train, y_train).predict(X_test)
            t2 = sklearn.metrics.accuracy_score(
                y_test, y_pred)
            avg.append(t2)
        avg2 = sum(avg)/len(avg)
        print(str(y*100) + "," + str(np.round(avg2, 5)))
    print(dd)

# Arbre seul
if True:
    dd = []
    t = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for x in t:
        avg = []
        for _ in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                output_corpus, na, test_size=0.2, random_state=0)
            arbreDesc = tree.DecisionTreeClassifier(max_depth=x)
            arbreDesc = arbreDesc.fit(X_train, y_train)
            y_pred = arbreDesc.predict(X_test)
            t2 = sklearn.metrics.accuracy_score(
                y_test, y_pred)
            avg.append(t2)
        avg2 = sum(avg)/len(avg)
        print(str(np.round(avg2, 5)))
        dd.append(avg2)
    plt.plot(t, dd)
    plt.ylabel("Précision")
    plt.xlabel("Profondeur max")
    plt.title("Arbre de décision 20%")
    plt.show()
# Forêt aléatoire

if False:
    dd = []
    t = [5, 25, 50, 100, 125,
         150, 175, 200, 225, 275, 300, 400, 500]
    for x in t:
        avg = []
        for _ in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                output_corpus, na, test_size=0.8, random_state=0)
            foretAl = RandomForestClassifier(n_estimators=x)
            foretAl = foretAl.fit(X_train, y_train)
            y_pred = foretAl.predict(X_test)
            t2 = sklearn.metrics.accuracy_score(
                y_test, y_pred)
            avg.append(t2)
        avg2 = sum(avg)/len(avg)
        print(str(x) + "," + str(np.round(avg2, 5)))
        dd.append(sum(avg)/len(avg))
    plt.plot(t, dd)
    plt.ylabel("Précision")
    plt.xlabel("Estimateurs")
    plt.title("Forêt aléatoire - 80%")
    plt.show()


# SVM
if False:
    x = [0.2, 0.5, 0.8]
    for y in x:
        avg = []
        for _ in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                output_corpus, na, test_size=y, random_state=0)
            supportVectMach = svm.SVC()
            supportVectMach = supportVectMach.fit(X_train, y_train)
            y_pred = supportVectMach.predict(X_test)
            t2 = sklearn.metrics.accuracy_score(
                y_test, y_pred)
            avg.append(t2)
        avg2 = sum(avg)/len(avg)
        print(str(np.round(avg2, 5)))

# MLP
if False:
    dd = []
    t = [50, 64, 78, 86, 95, 98, 100, 102, 105, 120, 150]

    for x in t:
        avg = []
        for _ in range(2):
            X_train, X_test, y_train, y_test = train_test_split(
                output_corpus, na, test_size=0.8, random_state=0)
            percMultCouche = MLPClassifier(
                hidden_layer_sizes=(x, x), random_state=0)
            percMultCouche = percMultCouche.fit(X_train, y_train)
            y_pred = percMultCouche.predict(X_test)
            t2 = sklearn.metrics.accuracy_score(
                y_test, y_pred)
            avg.append(t2)
        avg2 = sum(avg)/len(avg)
        print(str(x) + "," + str(np.round(avg2, 5)))
        dd.append(sum(avg)/len(avg))
    plt.plot(t, dd)
    plt.ylabel("Précision")
    plt.xlabel("Dimension")
    plt.title("MPL - 80%")
    plt.show()
