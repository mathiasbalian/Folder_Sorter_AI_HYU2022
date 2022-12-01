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
The tool would first have to undergo a phase of machine learning in order to gain in precision and to keep improving in categorizing files. We will use some word databases for various categories in order to decide the content subject of a file and give the AI the ability to categorize a large range of files.

## 3. The Bag Of Words method
This time, we used a very different approach than the previous one. As the first method didn't include some Machine Learning techniques and algorithms, we decided that it would be a good idea to approach our idea in a different way. That's why we used the Bag Of Words (BOW) method.

### The dataset
First of all, we need a dataset in order to train our ML model. For this, we downloaded some files (pdf and docx) on internet with some contents of the different school subjects that our project is able to handle. We then classified them into folders corresponding to the subject of each document. Each folder contained around 10 files, which were all about 10-15 pages long.
From this point, we had our dataset ready. The first thing that we had to do is creating a dataframe using the pandas library. For this, we first created a list containing all the data, i.e the text that was extracted form each file of each subject.
  
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
  
From now on, we could start treating the text that was extracted.  

### Text preprocessing
When reading and extracting text from a pdf or word document, the text can quickly become bloated with some unwanted escape characters, single characters, empty characters etc... That's why we need to tokenize and lemmatize the text. What does this mean ?
- **Text tokenization**  
Text tokenization is the process of separating a text into "tokens". Usually, we do this by splitting the text by whitespaces, removing escape characters and punctuation and putting the text in lowercase. Everything that could be considered as unwanted in a text where only the words themsevles are important is removed.  
If we take back our dataframe previously created, we can iterate over each text in this dataframe, and tokenize it using regex expressions:

```python
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

texts = list(dataframe["Content"])

for content in texts:
    clean = re.sub('[^a-zA-Z]', ' ', content)
    clean = re.sub(r'\W', ' ', content)  # Replace all escape characters (\n, \t, etc...)
    clean = re.sub(r'\s+[a-zA-Z]\s+', ' ', content)  # Replace single characters by a whitespace
    clean = re.sub(r'[^\w\s]', '', content)  # Replace all characters that are not a letter
    clean = clean.lower()  # Put the text in lowercase
    clean = clean.split()  # Split the text by whitespaces
    clean = [word for word in clean if word not in stopwords.words('english')]  # Remove all english stop words (the, an, in...)
```
For example the code above transforms the string   
"The\n .sky is 4beautiful, today! "   
into:
```console
['sky', 'beautiful', 'today']
```
As you can see, the text is cleaner and we got rid of all unwanted characters and useless words.  
The next step is to lemmatize the text  
- **Text lemmatization**  
Lemmatization is the process of switching any word to its base root. For example, "leaves" becomes "leaf". Similarly, "caring" becomes "care".
This is very important as it makes it much easier for our ML model to learn and understand the topics of the text that it is analysing.  
With the previous step and te lemmatization of the text, we get the following code for our text preprocessing:
```python
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
texts = list(dataframe["Content"])
texts_cleaned = []

for content in texts:
    clean = re.sub('[^a-zA-Z]', ' ', content)
    clean = re.sub(r'\W', ' ', content)
    clean = re.sub(r'\s+[a-zA-Z]\s+', ' ', content)
    clean = re.sub(r'[^\w\s]', '', content)
    clean = clean.lower()
    clean = clean.split()
    clean = [word for word in clean if word not in stopwords.words('english')]
    clean = [lemmatizer.lemmatize(word) for word in clean]
    clean = ' '.join(clean)
    texts_cleaned.append(clean)

dataframe['Content'] = texts_cleaned
```
We have now tokenized and lemmatized each text that we extracted and stored in our dataset.  
Now that this is finished, we can start buiding our ML model.

### Building the model



