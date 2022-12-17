import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

file = open("data.txt", "r")

data = []

c1 = 0

for l in file.readlines():
    if c1 == 10000:
        break

    if "$$" in l:
        continue

    content = l.split(" ")

    r = []  # [sens#, word1,word2,word3,word4,tword1,tword2,tword3,tword4]
    sentence = []
    index = -1
    c = 0
    interest = 0
    w = ""
    for t in content:
        if "=" in t:
            continue
        if t == "[" or t == "]":
            continue
        if t == "\n":
            continue

        tt = t.split("/")
        if len(tt) == 2:
            if "interest" in t and "_" in t:
                w = t
                interest = c
            sentence.append(tt)
            c += 1

    d = ["@" for x in range(9)]

    tt = sentence[interest][0].split("_")
    if len(tt) != 2:
        pass
    else:
        d[0] = tt[1]

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
    c1 += 1

t = []
a = []
for i in data:
    t.append(" ".join(i[1:]))
    a.append(i[0])


vectorizer = CountVectorizer(
    token_pattern=r"\b\w+\-\w+\b|\b\w+\.\w+\.\b|\b\w+\.\w+\b|\b\w+\b|\@|\,|\.|\%|\`\`")
X = vectorizer.fit_transform(t).toarray()
tokenizer = vectorizer.build_tokenizer()
output_corpus = []
na = []

for i, element in enumerate(t):
    tt = element
    element = tokenizer(element.lower())
    if(len(element) == 8):
        output_line = []
        for token in element:
            output_line.append(vectorizer.vocabulary_.get(token))
        output_corpus.append(output_line)
        na.append(a[i])
X_train, X_test, y_train, y_test = train_test_split(
    output_corpus, na, test_size=0.5, random_state=0)
gnb = MultinomialNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" %
      (len(X_test), (y_test != y_pred).sum()))
# Shit to do:
# 1 - Lire data for txt
# 2 - Information contextuels de la phrase: mots avant/ apres, catégories des mots autour, ...
# 3 -


# Naive Bayes

# Decision tree

# Random Forest

# SVM

# Multi Layers Perceptron
