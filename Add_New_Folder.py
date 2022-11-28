import os
import tkinter as tk
from tkinter import filedialog

def create_folder(subject, folder_path):
    path = os.path.join(folder_path, subject)
    try:
        os.mkdir(path)
        print('A folder named ', subject, ' has been created.')
    except FileExistsError:
        print('A folder named ', subject, ' already exists.')


subject = 'Biology'
tk.Tk().withdraw()
folder_path = filedialog.askdirectory()
create_folder(subject, folder_path)

