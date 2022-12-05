# Folder Sorter

## 1. Team members
- Mathias BALIAN | mathias.balian92@gmail.com | Hanyang University
- Manon GARDIN | manon.gardin@gmail.com | Hanyang University
- Apiram UDHAYALINGAM | apiram.udhayalingam@gmail.com | Hanyang University
  
## 2. Introduction
- **_Why are we doing this ?_**  
  
Have you ever had a folder somewhere in your pc that was a total mess, completely filled with unsorted files of all types and subjects ? You realized too late that it was a mess, and once you saw it, you quickly discovered that it would take you ages to sort this folder to make it look somewhat clean and tidy. We, the members of     this team, have often been confronted to this situation, and especially for school-related contents where every file is called 'Assignment 1', 'Assignment 2' etc,      without precising the corresponding subject. Forget just once to put a file in the right folder, and then we lose an immense time opening it to see what's inside. And  after accumulating a large amount of unsorted file, we just never clean up our folders. That's why we wanted to develop a tool preventing messy directories. We want an organized worskpace, not a total mess.  
  
- **_What do we want to see at the end ?_**  
  
Our goal is to have a functionning AI tool that would sort and organize our files and directories into different categories for us. This tool would read the content of our files in the designated folder, analyse it and proceed to sort the files into folders corresponding to the categories of the files, which would be school subjects (Chemistry, Mathematics, Philosophy...). 
Our first idea was to use some word databases for various categories in order to get the content subject of a file and give the AI the ability to categorize a large range of files.
But this idea didn't make us really satisfied. We prefered to make the tool having to undergo a phase of machine learning in order to gain in precision and to keep improving in categorizing files. We tried to do that way, and we chose to show you both of our works.

## 3. The rule-based method 

### The dataset 

For our first idea, we needed a dataset with as many words as possible per specific subject. If the tool needs to categorize a folder with both chemistry and computer science documents, it needs some chemistry and computer science related words in order to know which file belongs to which category. However, we couldn't find such datasets containing only words from a specific category, so we had to create our own. For that, we went on different websites listing the most used and common words related to a category, extracted the majority of the words that we thought were coherent, and wrote them all in a text file. In this file, each line contains all the words of a single category. Each word of a category is separated by a ";" which makes it easy to read in the code. At the end of a category, a new line is used. This allows us to read all te words of a category by a single call of the readline function in python.
  
![image](https://user-images.githubusercontent.com/107269689/204087677-db7d02de-1cd7-4ca2-9ba9-e6c4f799b59a.png)  
<sub>Each line corresponds to a school subject (in order: Biology, Computer science, Physics, Chemistry, Mathematics, Philosophy)</sub>  
  
In our code, we create a dictionary where the keys are the different school subjects and the values are the list of words from this school subject:

```python
f = open("Dataset_Topics.txt", "r")

# We create a dictionary where the key is a school subject
# and the value is the list of the words related to this subject and a counter related to the subject
dataset = {"biology": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "compsci": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "physics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "chemistry": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "mathematics": [list(dict.fromkeys(f.readline().split(";"))), 0],
           "philosophy": [list(dict.fromkeys(f.readline().split(";"))), 0]}
f.close()
```
### Methodology

Let's quickly explain the steps of our algorithm. 
As explained in the previous part, we first initialized our dataset as a dictionary.
Then, the users can choose the folder that they want to classify. We wanted this to be easy to understand for every users, so we display the file explorer so that they can directly click on the folder they want.
After that, everything is set up. The user will not interact with the code anymore. We're going to analyze the files one by one in the selected folder. 
Here are the steps for this purpose :
- We start to extract the content of the first file, which can be a .pdf, a .doc or a .docx
- We go through each word of the text, and for every subject, we check if the word appears in the list of words of the subject. If yes, then we increment the counter of the subject.
- Still for the same file, we determine the maximum counter to get the right subject.
- Finally, we create a folder that have the name of the right subject and we move the file in it. If the folder already exists, we directly move the file in it.

### The code

#### **_Text extraction_** 

We created a separate file named Util.py, which contains two functions concerning the text extraction.
Our ojective is to get the content of files that we want to sort, but we have to consider the format of the files. We chose to work with only .pdf and .docx files. 
Let's start with pdf files :
We used a library named PyPDF2 for the function 'textfrompdf(path)'. The function takes a string variable as parameter, which is the path to the pdf files in the file explorer. We had to take care of an exception which is "Unable to open the file", because sometimes pdf files are encrypted, or just are simply not supported. By using the library PyPDF2, we are able to read the content of the file with the function PyPDF2.PdfFileReader(file) and stock it in a string variable with the .extractText() function. Finally we don't forget to split the words by spaces.
Here is the code :

```python
def textfrompdf(path):
    fileobj = None
    try:
        fileobj = open(path, mode='rb')
    except:
        print("Unable to open the file")

    pdfcontent = ""  # The string containing the text of the file
    pdfreader = PyPDF2.PdfFileReader(fileobj)
    for i in range(0, pdfreader.numPages):  # We iterate over the pages of the document
        pageobj = pdfreader.getPage(i)
        pdfcontent += pageobj.extractText()

    return list(filter(None, re.split(r'[\r\n\t\xa0]+| ', pdfcontent)))
```

The concept for the docx files is the same as pdf, but this time we use the textract library. We're going to give details about libraries in a next part.

```python
def textfromword(path):
    # Test if the path is valid
    file = None
    try:
        file = open(path, "r")
    except:
        print("Unable to open the file")
    finally:
        file.close()

    text = textract.process(path)  # We extract the text from the file
    text = text.decode("utf-8")
    return list(filter(None, re.split(r'[\r\n\t\xa0]+| ', text)))
```

#### **_Text Analyzing_** 

Now, let's deal with the principal file FolderSorter.py. First, we open the file explorer with the library tkinter, which can allow the user to directly select the folder he wants. The function filedialog.askdirectory() return the path to the folder in a string variable.
```python
import tkinter
from tkinter import filedialog

tkinter.Tk().withdraw()
folder_path = filedialog.askdirectory()
```

Next, we don't forget to take care of the exception like when there is no file or no more file to process in the folder, and when the file is not supported or can't be open.
So, we start by extracting the text with the functions we explained just before. Then, we count the words related to each subject in the extracted text. For this purpose, we created a function called wordcounter(filetext). This function go through all words of the text and detect whether the word is in the dictionary and for which subject. If the word exists, the counter of the specific subject is incremented. 

```python
def wordcounter(filetext):
    global total_counter
    for word in filetext:
        word = word.lower()
        total_counter += 1
        for subject in dataset:
            if dataset[subject][0].count(word) > 0:
                dataset[subject][1] += 1
 ```
 
 After that, we can obtain the final category of the file by finding the maximum counter. We did that in a function called getsubject() which returns a string representing the name of the category result.
 
 ```python
 def getsubject():
    maximum = 0
    category = ""
    for subject in dataset:
        if dataset[subject][1] > maximum:
            maximum = dataset[subject][1]
            category = subject
    return category
 ```
 
 And then we reset the counters for the next file with this little function :
 
 ```python
 def resetcounters():
    for subject in dataset:
        dataset[subject][1] = 0
 ```
 
 Finally, with the library os, we create a folder named as the category we obtained, and with the library shutil, we move the file to this folder. We don't forget to check if the category file already exists or not. If it does, then we don't create a new one and we just move the file in the existing folder.
 
 Here is our final code :
 
 ```python
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
 ```

## 6. Related Work
### Prerequisites
As this project uses some libraries that are not included in the default python package, we need to install them manually. For this, simply run the following command:
  
```console
$ pip install -r requirements.txt
```

Like we explained in the previous parts, we needed the PyPDF2 library. We used its documentation : https://pypdf2.readthedocs.io/en/latest/user/extract-text.html.
We only used ths library to read the content of a PDF file, but it is capable of way more than that.
For the same goal but with docx documents, we used the textract library. Here is the documentation we used : https://textract.readthedocs.io/en/stable/.
We could've also used this library for PDF files, but we started with PyPDF2 first, so we didn't go back on our steps.

The tkinter library permits to create Graphical User Interfaces. We used it to display the file explorer so that the users can directly select the file they want to order. Here is the documentation : https://docs.python.org/3/library/tkinter.html

Finally, for our idea, we needed a way to interact with the operating system, which means we needed to create and move folders or files. For this purpose, the os library was adapted. The library let us check if a file exist or not, and work with directories. Here is the documentation : https://www.analyticsvidhya.com/blog/2021/05/30-useful-methods-from-python-os-module/
In the same purpose, the shutil library also allows us to interact with files, we used it to move the files from one folder to an other. https://docs.python.org/3/library/shutil.html

## 7. Conclusion

(Talk about the limits like no permission to move a file)
