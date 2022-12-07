import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from Util import textfrompdf, textfromword
import os
import pickle
import tkinter
import shutil
from tkinter import filedialog
from Train_Model import cv

modeload = open("ML_model", "rb")
model = pickle.load(modeload)
modeload.close()

# We open a file dialog for the user
folder_path = filedialog.askdirectory()

try:
    for filename in os.listdir(folder_path):
        file = os.path.join(folder_path, filename)
        if os.path.isfile(file):
            file_text = ""
            if os.path.splitext(file)[1] == ".pdf":  # If the file is a pdf file
                file_text = textfrompdf(file)
            elif os.path.splitext(file)[1] == ".docx":  # If the file is a docx or doc
                file_text = textfromword(file)

            subject = model.predict(cv.transform([file_text]))[0]
            try:
                os.mkdir(folder_path + "/" + subject)
            except:
                pass

            shutil.move(file, folder_path + "/" + subject)
        else:
            print("The file", file, "is not supported.")

except:
    print("File not provided or not found")