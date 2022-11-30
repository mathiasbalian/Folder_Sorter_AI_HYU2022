import os
import tkinter
import shutil
from tkinter import filedialog
from Util import textfrompdf, textfromword

f = open("Dataset_Topics.txt", "r")

# We create a dictionary where the key is a school subject
# and the value is the list of the words related to this subject and a counter for the subject
dataset = {"Biology": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Compsci": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Physics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Chemistry": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Mathematics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Philosophy": [list(dict.fromkeys(f.readline().split(";"))), 0]}
f.close()


def wordcounter(filetext):
    global total_counter
    for word in filetext:
        word = word.lower()
        for subject in dataset:
            if dataset[subject][0].count(word) > 0:
                dataset[subject][1] += 1


def getsubject():
    maximum = 0
    category = ""
    for subject in dataset:
        if dataset[subject][1] > maximum:
            maximum = dataset[subject][1]
            category = subject
    return category


def resetcounters():
    for subject in dataset:
        dataset[subject][1] = 0


# We open a file dialog for the user
tkinter.Tk().withdraw()
folder_path = filedialog.askdirectory()

try:
    for filename in os.listdir(folder_path):
        total_counter = 0
        file = os.path.join(folder_path, filename)
        if os.path.isfile(file):
            text = []
            if os.path.splitext(file)[1] == ".pdf":  # If the file is a pdf file
                text = textfrompdf(file)
            elif os.path.splitext(file)[1] == ".doc" or os.path.splitext(file)[1] == ".docx":  # If the file is a docx or doc
                text = textfromword(file)

            wordcounter(text)
            category = getsubject()
            resetcounters()

            try:
                os.mkdir(folder_path + "/" + category)
            except:
                pass

            shutil.move(file, folder_path + "/" + category)

        else:
            print("The file", file, "is not supported.")

except:
    print("File not provided or not found")

