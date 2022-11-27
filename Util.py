import PyPDF2
import textract
import re


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

    return list(filter(None, re.split(r'[\r\n\t\xa0]+| ', pdfcontent)))
    # Note on the return above: here, we split the text extracted from the file
    # using many parameters, first: \n, \r, \t, \xa0 . This is done because when
    # reading the content of a pdf or word file, a lot of these escape characters can appear
    # in the string, which is something that we don't want. Then, we separate every word
    # by the space characters between them. Finally, we filter the resulting list from the split
    # to remove any None element in the list, cast this filter to a list and return the list.


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
    return list(filter(None, re.split(r'[\r\n\t\xa0]+| ', text)))
