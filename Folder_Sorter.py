import os
import tkinter
from tkinter import filedialog
from Util import textfrompdf, textfromword

f = open("Dataset_Topics.txt", "r")

# We create a dictionary where the key is a school subject
# and the value is the list of the words related to this subject
dataset = {"biology": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "compsci": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "physics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "chemistry": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "mathematics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "philosophy": [list(dict.fromkeys(f.readline().split(";"))), 0]}
f.close()

# Counters for each school subject possible
biology_counter = 0
compsci_counter = 0
physics_counter = 0
chemistry_counter = 0
philosophy_counter = 0
total_counter = 0

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

            for word in text:  # We iterate over each word in the file's text
                word = word.lower()  # Put it to lowercase to match dataset words
                total_counter += 1
                for subject in dataset:  # We iterate over each subject of the dataset
                    if dataset[subject][0].count(word) > 0:  # If the word from the text is present is the subject's list of words
                        dataset[subject][1] += 1

        else:
            print("The file", file, "is not supported.")

except:
    print("File not provided or not found")

for subject in dataset:
    print(subject, dataset[subject][1])
