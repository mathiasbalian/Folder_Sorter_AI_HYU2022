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

## 3. The Dataset
In order for our tool to classify the files, it needs to rely on a set of words related to a specific subject. If the tool needs to categorize a folder with both chemistry and computer science documents, it needs some chemistry and computer science related words in order to know which file belongs to which category. However, we couldn't find such datasets containing only words from a specific category, so we had to create our own. For that, we went on different websites listing the most used and common words related to a category, extracted the majority of the words that we thought were coherent, and wrote them all in a text file. In this file, each line contains all the words of a single category. Each word of a category is separated by a ";" which makes it easy to read in the code. At the end of a category, a new line is used. This allows us to read all te words of a category by a single call of the readline function in python.
  
![image](https://user-images.githubusercontent.com/107269689/204087677-db7d02de-1cd7-4ca2-9ba9-e6c4f799b59a.png)  
<sub>Each line corresponds to a school subject (in order: Biology, Computer science, Physics, Chemistry, Philosophy)</sub>  
  
In our code, we create a dictionary where the keys are the different school subjects and the values are the list of words from this school subject:

```python
f = open("Dataset_Topics.txt", "r")

dataset = {"biology": list(dict.fromkeys(f.readline().split(";"))),
           "compsci": list(dict.fromkeys(f.readline().split(";"))),
           "physics": list(dict.fromkeys(f.readline().split(";"))),
           "chemistry": list(dict.fromkeys(f.readline().split(";"))),
           "philosophy": list(dict.fromkeys(f.readline().split(";")))}
           
f.close()
```

