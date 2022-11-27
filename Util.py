import PyPDF2
import os


# Function which takes a pdf file path and returns the text in the file
def textfrompdf(path):
    fileobj = None
    try:
        fileobj = open(path, mode='rb')
    except:
        print("Unable to open the file")

    pdfreader = PyPDF2.PdfFileReader(fileobj)
    pageobj = pdfreader.getPage(0)
    return pageobj.extractText()

