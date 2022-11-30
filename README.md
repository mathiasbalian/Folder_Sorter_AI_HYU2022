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

## 3. The Dataset
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
##4. Methodology
First, let's deal with our first idea, which was to use a dataset of words to classify the files. 
As explained in the previous part of the blog, we first initialized our dataset as a dictionary.
Then, the users can choose the folder that they want to classify. We wanted this to be easy to understand for every users, so we display the file explorer so that they can directly click on the folder they want.
After that, everything is set up. The user will not interact with the code anymore. We're going to analyze the files one by one in the selected folder. 
Here are the steps for this purpose :
- We start to extract the content of the first file, which can be a .pdf, a .doc or a .docx
- We go through each word of the text, and for every subject, we check if the word appears in the list of words of the subject. If yes, then we increment the counter of the subject.
- Still for the same file, we determine the maximum counter to get the right subject.
- Finally, we create a folder that have the name of the right subject and we move the file in it. If the folder already exists, we directly move the file in it.

## 6. Related Work
### Prerequisites
As this project uses some libraries that are not included in the default python package, we need to install them manually. For this, simply run the following command:
  
```console
$ pip install -r requirements.txt
```
