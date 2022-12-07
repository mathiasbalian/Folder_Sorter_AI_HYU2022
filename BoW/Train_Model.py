from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Util import textfrompdf, textfromword
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import tkinter
import shutil
from tkinter import filedialog


data = []

for subdirs, dirs, files in os.walk("Documents"):
    for file in files:
        if os.path.splitext(file)[1] == ".pdf":
            data.append([subdirs[10:], textfrompdf(os.path.join(subdirs, file))])
        elif os.path.splitext(file)[1] == ".docx":
            data.append([subdirs[10:], textfromword(os.path.join(subdirs, file))])

dataframe = pd.DataFrame(data, columns=["Subject", "Content"])

Texts = dataframe["Content"]
Subjects = dataframe["Subject"]

cv = CountVectorizer()
Texts = cv.fit_transform(Texts)

Texts_train, Texts_test, Subjects_train, Subjects_test = train_test_split(Texts, Subjects, test_size=0.5, random_state=0)

classifier = RandomForestClassifier()
classifier.fit(Texts_train, Subjects_train)
prediction = classifier.predict(Texts_test)

pickle.dump(classifier, open("ML_model", 'wb'))


