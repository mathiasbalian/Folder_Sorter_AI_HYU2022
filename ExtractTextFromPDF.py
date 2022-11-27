import PyPDF2

pdfFileObj = open('pdf-exemple.pdf', mode='rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
print(pageObj.extractText())
