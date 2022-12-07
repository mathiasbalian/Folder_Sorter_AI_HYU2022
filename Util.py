import PyPDF2
import textract
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Function that tokenizes and lemmatizes the text
def tokenize_lemmatize(text):
    lemma = WordNetLemmatizer()
    clean_text = re.sub(r'\W', ' ', str(text))  # Remove all escape characters
    clean_text = re.sub('[^a-zA-Z]', ' ', str(text))  # Remove all single characters
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [word for word in clean_text if word not in stopwords.words('english')]
    clean_text = [lemma.lemmatize(word) for word in clean_text]
    clean_text = ' '.join(clean_text)
    return clean_text


# Function which takes a pdf file path and returns the text in the file
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


# Function which takes a word (doc or docx) file path and returns the text in the file
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


