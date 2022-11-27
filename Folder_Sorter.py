import os
import tkinter
from tkinter import filedialog
from Util import textfrompdf

f = open("Dataset_Topics.txt", "r")

# We create a dictionary where the key is a school subject
# and the value is the list of the words related to this subject
dataset = {"biology": list(dict.fromkeys(f.readline().split(";"))),
           "compsci": list(dict.fromkeys(f.readline().split(";"))),
           "physics": list(dict.fromkeys(f.readline().split(";"))),
           "chemistry": list(dict.fromkeys(f.readline().split(";"))),
           "philosophy": list(dict.fromkeys(f.readline().split(";")))}
f.close()

# We open a file dialog for the user
tkinter.Tk().withdraw()
folder_path = filedialog.askdirectory()

try:
    for filename in os.listdir(folder_path):
        file = os.path.join(folder_path, filename)
        if os.path.isfile(file):
            file_extension = os.path.splitext(file)[1]
            print(file_extension)
except:
    print("File not provided or not found")
