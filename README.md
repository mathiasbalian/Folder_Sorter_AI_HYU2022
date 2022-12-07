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
In addition of this rule-based, we decided to use other methods using known Machine Learning algorithms. We tried two other techniques: the Bag Of Words method and the Neural Network.

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
    fileobj.close()
    
    return tokenize_lemmatize(pdfcontent)
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
    return tokenize_lemmatize(text)
```  
Notice that in both functions, we return a variable defined by the tokenize_lemmatize function. Let's see what this function does.

#### **_Text preprocessing_**
When reading and extracting text from a pdf or word document, the text can quickly become bloated with some unwanted escape characters, single characters, empty characters etc... That's why we need to tokenize and lemmatize the text. What does this mean ?
- **Text tokenization**  
Text tokenization is the process of separating a text into "tokens". Usually, we do this by splitting the text by whitespaces, removing escape characters, punctuation, useless words and putting the text in lowercase. Everything that could be considered as unwanted in a text where only the words themsevles are important is removed.  

```python
import re
from nltk.corpus import stopwords

def tokenize_lemmatize(text):
    lemma = WordNetLemmatizer()
    clean_text = re.sub(r'\W', ' ', str(text))  # Remove all escape characters
    clean_text = re.sub('[^a-zA-Z]', ' ', str(text))  # Remove all single characters that are not letters
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [word for word in clean_text if word not in stopwords.words('english')]
```
Here, what this piece of code does is that it removes all single characters and escape characters by a whitespace from the text in the function arguments, puts the text in lowercase and splits the text by whitespaces. Finally, we remove all english stopwords in the text. These stopwords represent some words in the english vocabulary that are not relevant to classify our documents (the, an, in...). For example the code above transforms the string   
"The\n .sky is 4beautiful, today! "   
into:
```console
sky beautiful today
```
As you can see, the text is cleaner and we got rid of all unwanted characters and useless words.  
The next step is to lemmatize the text  
- **Text lemmatization**  
Lemmatization is the process of switching any word to its base root. For example, "leaves" becomes "leaf". Similarly, "caring" becomes "care".
With the previous step and the lemmatization of the text, we get the following code for our text preprocessing:
```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def tokenize_lemmatize(text):
    lemma = WordNetLemmatizer()
    clean_text = re.sub(r'\W', ' ', str(text))  # Remove all escape characters
    clean_text = re.sub('[^a-zA-Z]', ' ', str(text))  # Remove all single characters that are not letters
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [word for word in clean_text if word not in stopwords.words('english')]
    clean_text = [lemma.lemmatize(word) for word in clean_text]
    clean_text = ' '.join(clean_text)
    return clean_text
```  
Same thing, the code above transforms the sting  
"The\n .sky is 4beautiful, these days1! "  
into
```console
sky beautiful day
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
    for word in filetext.split(' '):
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
        file = os.path.join(folder_path, filename)
        if os.path.isfile(file):
            text = []
            if os.path.splitext(file)[1] == ".pdf":  # If the file is a pdf file
                text = textfrompdf(file)
            elif os.path.splitext(file)[1] == ".docx":  # If the file is a docx or doc
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
 
 ## 4. The Bag Of Words model
This time, we used a very different approach than the previous one. As the first method didn't include some Machine Learning techniques and algorithms, we decided that it would be a good idea to approach our idea in a different way. That's why we used the Bag Of Words (BOW) method.

### The dataset
First of all, we need a dataset in order to train our ML model. For this, we downloaded some files (pdf and docx) on internet with some contents of the different school subjects that our project is able to handle. We then classified them into folders corresponding to the subject of each document. Each folder contained around 10 files, which were all about 10-15 pages long.
From this point, we had our dataset ready. The first thing that we had to do is creating a dataframe using the pandas library. For this, we first created a list containing all the data, i.e the text that was extracted from each file of each subject.
  
```python
import pandas as pd

data = []

for subdirs, dirs, files in os.walk("Documents"):  # Iterate over each sudirectory of the "Documents" directory
    for file in files:  # Iterate over each file of the current subdirectory
        if os.path.splitext(file)[1] == ".pdf":  # If the document is a pdf file
            data.append([subdirs[10:], textfrompdf(os.path.join(subdirs, file))])
        elif os.path.splitext(file)[1] == ".docx":  # If the document is a docx file
            data.append([subdirs[10:], textfromword(os.path.join(subdirs, file))])

dataframe = pd.DataFrame(data, columns=["Subject", "Content"])
```  
Everytime the textfrompdf or textfromword functions are called, the text appended to the "data" list is already preprocessed, meaning it has already been tokenized and lemmatized. Therefore, it is ready to be treated by our ML model.

### Building the model
Now that the text has been processed and adapted for a Machine Learning algorithm, we can begin to create our model. As previously mentionned, we will be using the Bag Of Words model. 
- **How does the Bag Of Words (BOW) model works ?**  
The BOW model is a way of representing the data of a text as numeric features. For this, the algorithm creates a vocabulary of the known words that occur in the documents' text that it is analyzing and then creates a vector for each document that contains as values the counters of how often the words from the vocabulary appear.
  
  
To use this model, we first extract the two columns of our dataframe: the subject of a document a the text that has been extracted from this document:
```python
Texts = dataframe["Content"]
Subjects = dataframe["Subject"]
```  
As mentioned above, we now need to tranform our text into a vector of numbers so that the ML algorithm can understand what it has to learn. We will use the CountVectorizer class included in the sklearn.feature_extraction.text library. We first start by declaring a CountVectorizer oject, and then we use the fit_transform function wich vectorizes our Text list.  
```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
Texts = cv.fit_transform(Texts)
```  

As we want to use a part of our dataset (the content of our dataframe) as a set for training the ML model, and the other part for testing the model, we need to find a way to split it into those two parts. For this, we will use the train_test_split function from the sklearn.model_selection library. This function does exactly what we want: it splits the dataset in a traning set and a testing set.
```python
from sklearn.model_selection import train_test_split

Texts_train, Texts_test, Subjects_train, Subjects_test = train_test_split(Texts, Subjects, test_size=0.5, random_state=0)
```
We pass into the function 4 parameters:
- Texts: a list containing all the texts extracted from the documents that have been vectorized.
- Subjects: a list of all the subjects of the texts contained in the Texts variable.
- test_size: The percentage of the dataset that we want to use as a test set. Here, we set the parameter as 0.5, which means that we use 50% of our dataset as a training set for our model.
- random_state: We set this parameter at 0. This means that the function will not spit our dataset randomly. If we would heva put any other values, we would get a different result each time because the function would split our dataset randomly.  
  
This function retruns 4 variables: 
- Texts_train: a numpy array containing all the vectorized texts used to train the model
- Texts_test: a numpy array containing all the vectorized texts used to test the model
- Subjects_train: a numpy array containing all the subjects used to train the model
- Subjects_test: a numpy array containing all the subjects used to test the model  
  
Once we have these different sets, we can now use a Machine Learning classifier to train with ur training sets and then test it with our test sets. There are many classifiers available for this, but we chose to use the RandomForestClassifier from the sklearn.ensemble library:
```python
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(Texts_train, Subjects_train)
```  
The fit function builds a forest of trees from the training set Texts_train, Subjects_train.
Once we have our classifier ready, we can now test it on our test sets, using the predict function, and print the accuracy of the classifier's prediction using the accuracy_score function from the sklearn.metrics library:
```python
from sklearn.metrics import accuracy_score

prediction = classifier.predict(Texts_test)
print("Accuracy: ", accuracy_score(Subjects_test, prediction))
```  
With all of this, our model gave us the following accuracy:
```console
Accuracy: 0.9
```  
This means that our model has around 90% accuracy, which is pretty decent! Of course, if we tweak the parameters of the train_test_split function and change the test_size parameter, we will get different results.  
We can now proceed to save the model using the Pickle library:
```python
import pickle

pickle.dump(classifier, open("ML_model", 'wb'))
```
### Using the model
Once we have saved the model, we can now create a new file, Sort_Folder.py. In this file, we will be able to call the model and choose the directory that we want to sort. We start by loading our model:
```python
import pickle

modeload = open("ML_model", "rb")
model = pickle.load(modeload)
modeload.close()
```  
Once this is done, we open a file dialog in order for the user to choose the directory that he wants to sort. After that, we iterate over each file of the folder and then extract the text.
```python
import os
from tkinter import filedialog
from Util import textfrompdf, textfromword

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
                
        else:
            print("The file", file, "is not supported.")
except:
    print("File not provided or not found")
```  
Then, once this is done, we can finally start calling the model that we previously trained and tested in the Train_Model.py file. From this file, we also use the same CountVectorizer() object that we used to train the model and that was called cv.
```python
import os
from Train_Model import cv

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
            
        else:
            print("The file", file, "is not supported.")
except:
    print("File not provided or not found")
```  
The model.predict function will return, for example, the following for a random file:
```console
["Compsci"]
```
That's why we have to get the element at the first index: because model.predict returns a list of the predictions made from the input.
Once we have te subject, we can proceed just like in the Rule-Based method, by trying to create a folder named after the subject, and moving the file to this new folder.
```python
import os
import shutil

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
```
As a result, we will get, as expected, a sorted folder, with subfolders of the school subjects found, and the files correctly sorted in those subfolders.
Of course, as our model isn't perfectly accurate, some files can be placed in the wrong folder. However, as our model has a 90% accuracy, this is not very likely to happen.

## 5. Neural Network model
Here's another model with Neural Network.

### The dataset
As for the Bag Of Words model, we need a dataset in order to train our NN model. For this, we downloaded some files (pdf only) on internet with some contents of the different school subjects that our NN is able to handle (Link where 95% of the files come from: https://ocw.mit.edu/). So, that our NN perform on our files, we decided to put some kind of score on each files. The score will be a list where each value is the number of words related to a topic. We perform on 5 topics, so the length of our list will be 5.
For example, if we have a file with this (very short) content:
```txt
Is a computer science and physics double major a good idea if I want to go to medical school?
```
The score will be [0, 2, 1, 0, 1] because 'computer' and 'double' are in computer science in our dataset, 'physics' in physics and 'science' in philosophy. So that our AI perform well, we need to have a good dataset.
Now that we have our files, we need to create a training set, a validation set and a testing set.

All our files are in a folder called 'FileForTraining' and because we want to do supervised learning, we decide labelised the files. In the 'Dataset_filename-Topics.csv', you can find two columns, the first one is for the name of the files and second one is to classify the file by topic. Here's the table for the classification:
| Topic | Value |
| --- | --- |
| Biology | 0 |
| Computer Science | 1 |
| Physics | 2 |
| Chemistry | 3 |
| Philosophy | 4 |

In the code below, we read, scan and score each files and we store them in the dataframes.
```python
key = ['biology', 'compsci', 'physics', 'chemistry', 'philosophy']
idx = dict()
for i in range(0,len(key)):
    idx[key[i]] = i

# Path towards the folder where there are all files
folder_path = os.path.abspath(os.getcwd()) + '\FileForTraining'

# For each file, we will count
scores = list()
data_filename_topics = pd.read_csv('Dataset_fileName-Topics.csv')
for filename, _ in tqdm(data_filename_topics.values):
    file = os.path.join(folder_path, filename)
    if(os.path.isfile(file)):
        text = None
        extension = os.path.splitext(file)[1]
        if extension == ".pdf":  # If the file is a pdf file
            with open(file, 'rb') as pdfFileObj:
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict = False)
                text = re.sub(r'[^\w\s]', ' ', pdfReader.getPage(0).extractText())
                for pageNumber in range(1, pdfReader.numPages):
                    pageText = re.sub(r'[^\w\s]', ' ', pdfReader.getPage(pageNumber).extractText())
                    text = ' '.join([text, pageText])

                text = text.split(' ')

        # If the file is a pdf, we can compute his score
        if text != None:
            score = np.zeros(len(key))
            for word in text:
                w = word.lower()
                for subject in dataset:
                    if(w in dataset[subject]):
                        score[idx[subject]] += 1
            scores.append(score)
    else:
        print("The file", file, "is not supported.")

# We decide to put all those information in dataframe
df_x = pd.DataFrame(np.array(scores), columns = key)
df_y = data_filename_topics['topic']
```
It took some times to read, scan and score all the files (3 minutes with our computers).
Here's what our dataframe give us:
```txt
     biology  compsci  physics  chemistry  philosophy
0        0.0     21.0      0.0        4.0         1.0
1        0.0     52.0      1.0        2.0         2.0
2        1.0     98.0      1.0        3.0         6.0
3        2.0    144.0      7.0        2.0        11.0
4        0.0    143.0      3.0        4.0        10.0
..       ...      ...      ...        ...         ...
299    161.0     51.0     23.0       51.0        20.0
300    615.0    256.0    209.0      115.0       150.0
301     46.0    688.0   1763.0      671.0      1126.0
302      3.0      7.0      2.0        2.0         7.0
303      5.0    103.0    258.0       63.0       277.0

[304 rows x 5 columns]
0      1
1      1
2      1
3      1
4      1
      ..
299    0
300    0
301    2
302    4
303    2
Name: topic, Length: 304, dtype: int64
```
### Let's analyse a bit the data
```python
for i in range(0, len(key)):
    print(f"{key[i]}: {len(df_y[df_y == i])}")
```
```txt
biology: 48
compsci: 83
physics: 96
chemistry: 46
philosophy: 31
```
The number of files per topics isn't really well balanced but we'll work with that. Before using our NN, let's see, if we sort according to the greatest number of words in a topic, if this corresponds to the related topic.

```python
prediction_max = np.array([np.argmax(row) for row in df_x.values])
print(f'Number of correct: {sum(df_y.values == prediction_max)} on {len(df_y)} ({sum(df_y.values == prediction_max)*100/len(df_y):.4} %)')
```
```txt
Number of correct: 189 on 304 (62.17 %)
```
It seems that just selecting the topic with the most commun words doesn't always work.

### Split the data

Now that we have our dataframe, we have to split it into 3 sets: training, validation, testing set.

```python
from sklearn.model_selection import train_test_split
import torch as t

# Split the data into 70% for training, 15% for validation and 15% for testing
train_x, rest_x, train_y, rest_y = train_test_split(df_x.values, df_y.values, train_size=0.7, shuffle=True)
val_x, test_x, val_y, test_y = train_test_split(rest_x, rest_y, train_size=0.5, shuffle=True)

# Transformation and normalization
train_x = t.tensor(train_x, dtype = t.float32)
val_x = t.tensor(val_x, dtype = t.float32)
test_x = t.tensor(test_x, dtype = t.float32)

train_y = t.tensor(train_y, dtype= int)
val_y = t.tensor(val_y, dtype= int)
test_y = t.tensor(test_y, dtype= int)
```

### Create the Neural Network

We'll use pytorch to create our NN. Our NN will have 3 hidden layers where each one will have 256 units and for the activation, we'll use ReLU and for the ouput LogSoftMax.

```python
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(MLP, self).__init__()

    # Inputs to hidden layer linear transformation
    self.input = nn.Linear(D_in, H)
    self.hidden = nn.Linear(H, H)
    self.hidden2 = nn.Linear(H,H)
    self.output = nn.Linear(H, D_out)

  def forward(self, x):
    x = F.relu(self.input(x))
    x = F.relu(self.hidden(x))
    x = F.relu(self.hidden2(x))
    y_pred = self.output(x)

    return y_pred
```

We have to create the training, the validation (to check loss on validation set) and the evaluation (to compute accuracy) function.

```python
def train_model(model, criterion, optimizer, train_x, train_y, val_x, val_y, num_epochs = 10, batch_size = 64, show_info = False):
  # Set model to train mode
  model.train()

  # Training loop
  for epoch in range(0,num_epochs):
    perm = t.randperm(len(train_y))
    sum_loss = 0.

    for i in range(0, len(train_y), batch_size):
      x1 = Variable(train_x[perm[i:i + batch_size]], requires_grad=False)
      y1 = Variable(train_y[perm[i:i + batch_size]], requires_grad=False)

      # Reset gradient
      optimizer.zero_grad()
      
      # Forward
      fx = model(x1)
      loss = criterion(fx, y1)
      
      # Backward
      loss.backward()
      
      # Update parameters
      optimizer.step()
      
      sum_loss += loss.item()

    val_loss = validation_model(model, criterion, val_x, val_y, batch_size)
    if(show_info and epoch%10==0):
      print(f"Epoch: {epoch} \tTraining Loss: {sum_loss} \tValidation Loss: {val_loss}")

def validation_model(model, criterion, val_x, val_y, batch_size):
  valid_loss = 0
  perm = t.randperm(len(val_y))

  # Set to validation mode
  model.eval()
  
  for i in range(0, len(val_y), batch_size):
      x1 = Variable(val_x[perm[i:i + batch_size]], requires_grad=False)
      y1 = Variable(val_y[perm[i:i + batch_size]], requires_grad=False)
      
      # Forward
      fx = model(x1)
      loss = criterion(fx, y1)
      
      valid_loss += loss.item()

  return valid_loss

def evaluate_model(model, test_x, test_y):
  model.eval()
  y_pred = model(test_x)

  y_pred = t.max(y_pred,1).indices
  accuracy = t.sum(y_pred == test_y)/len(y_pred)
  
  return accuracy
```

### Train the model and check it's accuracy

After several training, we finally chose these hyperparameters.
```python
# Hyperparameters
learning_rate = 1e-3
epochs = 100
batch_size = 8

D_in, H, D_out = train_x.shape[1], 256, len(key) # D_in is the number of parameters (so 5 for us); D_out is the number of classes (so 5 for us)
model = MLP(D_in, H, D_out)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train the model
train_model(model, criterion, optimizer, train_x, train_y,
            val_x, val_y, epochs, batch_size, show_info = True)

#Evaluate the model
accuracy = evaluate_model(model, test_x, test_y)*100
print(f'Accuracy: {accuracy} %')
```
```txt
Epoch: 0 	Training Loss: 48.503595769405365 	Validation Loss: 14.585851907730103
Epoch: 10 	Training Loss: 31.11023785918951 	Validation Loss: 8.923621118068695
Epoch: 20 	Training Loss: 11.212598226964474 	Validation Loss: 4.466354936361313
Epoch: 30 	Training Loss: 9.06130476295948 	Validation Loss: 5.072529688477516
Epoch: 40 	Training Loss: 7.096194840967655 	Validation Loss: 5.090612441301346
Epoch: 50 	Training Loss: 22.226621463894844 	Validation Loss: 7.405122563242912
Epoch: 60 	Training Loss: 6.410949652083218 	Validation Loss: 7.692902863025665
Epoch: 70 	Training Loss: 11.587390199303627 	Validation Loss: 5.829236976802349
Epoch: 80 	Training Loss: 3.534953062655404 	Validation Loss: 7.492872446775436
Epoch: 90 	Training Loss: 3.3155347015708685 	Validation Loss: 8.136256963014603
Accuracy: 82.60869598388672 %
```
As we see, we have an accuracy of 82.61%. It seems pretty good for us. If we don't have a higher accuracy, it could be because, we don't have enough files to train our model or maybe because our dataset for the scoring is not good (not enough words in the topics, ...).

### Analyse the mistake made by the model

```python
y_pred = model(test_x)
y_pred = t.max(y_pred,1).indices

key2 = key.copy()
key2.append('Total')
df_result = pd.DataFrame(np.zeros((len(key),len(key) + 1), dtype= int), columns = key2,  index = key)
df_test_y = pd.DataFrame(test_y, dtype = int)
df_y_pred = pd.DataFrame(y_pred, dtype = int)
for i in range(0,len(key)):
    l = df_test_y[df_y_pred[0] == i]
    df_result.values[i][len(key)] = len(l)
    for j in range(0,len(key)):
        df_result.values[i][j] = len(l[l[0] == j])

print(df_result)
```
```txt
            biology  compsci  physics  chemistry  philosophy  Total
biology           4        0        0          1           0      5
compsci           0       12        3          0           0     15
physics           0        0       13          1           1     15
chemistry         0        0        1          4           0      5
philosophy        1        0        0          0           5      6
```
In this table, each row represents a kind of file that the neural network should initially associate. The columns represent the number of files that the neural network had associated to a subject. Even if our testing set isn't large, thanks to this table, we can observe that some computer science files could be mistaken with physics files.

So maybe, we have a problem with our dataset.

### Save the model

The model will be save in the file 'save_model.pt'.

```python
save_model_path = os.path.abspath(os.getcwd()) + '/save_model.pt'
t.save(model, save_model_path)
```

### Use the model

Firstly you have to load the model.

```python
import os
import numpy as np
import torch as t
from tkinter import Tk, filedialog
import shutil
from tqdm import tqdm

load_model_path = os.path.abspath(os.getcwd()) + r'\save_model.pt'
load_model = t.load(load_model_path)
load_model.eval()
```

After that you have to select the directory where are stored the file, you need to sort and the program will scan all the files to create data for the model. The scan can take some times.

```python
# This the list of all topics that our AI knows and was train for
key = ['biology', 'compsci', 'physics', 'chemistry', 'philosophy']

root = Tk()
root.withdraw()

root.attributes('-topmost', True)
folder_path = filedialog.askdirectory()

# For each file, we will count
scores = list()
filename_list = list()
print("Scan of the files in progress...\t(can take some times with several files)")
for filename in tqdm(os.listdir(folder_path)):
    file = os.path.join(folder_path, filename)
    if(os.path.isfile(file)):
        text = None
        extension = os.path.splitext(file)[1]
        if extension == ".pdf":  # If the file is a pdf file
            with open(file, 'rb') as pdfFileObj:
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict = False)
                text = re.sub(r'[^\w\s]', ' ', pdfReader.getPage(0).extractText())
                for pageNumber in range(1, pdfReader.numPages):
                    pageText = re.sub(r'[^\w\s]', ' ', pdfReader.getPage(pageNumber).extractText())
                    text = ' '.join([text, pageText])

                text = text.split(' ')

        # If the file is a pdf, we can compute his score
        if text != None:
            score = np.zeros(len(key))
            for word in text:
                w = word.lower()
                for subject in dataset:
                    if(w in dataset[subject]):
                        score[idx[subject]] += 1
            scores.append(score)
            filename_list.append(file)
    else:
        print("The file", file, "is not supported.")
print("Scan finish")
```
Finally, you run the code below to sort all you files. The program will move all your files and sorted them by creating folder for each topics. These folder will be created in the same directory, selected before.

```python
# Path where the AI will create the folder with the files sorted
# The result will appear in the same folder that you have selected but if you want, you can modify the path
files_sorted_path = folder_path

data = t.tensor(np.array(scores), dtype = t.float32)

# We predict to which topic are related each files
load_model.eval()
data_prediction = load_model(data)
data_prediction = t.max(data_prediction,1).indices

# We create the folder of the topics found by the AI
for i in range(len(key)):
    if i in data_prediction:
        if not os.path.exists(files_sorted_path + f'/{key[i]}'):
            os.makedirs(files_sorted_path + f'/{key[i]}')

# We move all the files in the topic sorted by the AI
for i in range(len(filename_list)):
    shutil.move(filename_list[i], files_sorted_path + f"/{key[data_prediction[i]]}/{os.path.basename(filename_list[i])}")
```

## 6. Related Work
### Prerequisites
As this project uses some libraries that are not included in the default python package, we need to install them manually. For this, simply run the following command:
  
```console
$ pip install -r requirements.txt
```

## 7. Conclusion

(Talk about the limits like no permission to move a file)
