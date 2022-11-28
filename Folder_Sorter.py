import os
import tkinter
from tkinter import filedialog
from Util import textfrompdf, textfromword, move_to_folder

f = open("Dataset_Topics.txt", "r")

# We create a dictionary where the key is a school subject
# and the value is the list of the words related to this subject
dataset = {"Biology": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Compsci": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Physics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Chemistry": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Mathematics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "Philosophy": [list(dict.fromkeys(f.readline().split(";"))), 0]}
f.close()

total_counter = 0

# We open a file dialog for the user
tkinter.Tk().withdraw()
folder_path = filedialog.askdirectory()

try:
    for filename in os.listdir(folder_path):
        for subject in dataset:
            dataset[subject][1] = 0
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

            max = 0
            subject_result = ""
            for subject in dataset:
                if dataset[subject][1] >= max:
                    max = dataset[subject][1]
                    subject_result = subject
            print(subject_result)
            move_to_folder(subject_result, folder_path, filename)


        else:
            print("The file", file, "is not supported.")

except:
    print("File not provided or not found")

for subject in dataset:
    print(subject, dataset[subject][1])
