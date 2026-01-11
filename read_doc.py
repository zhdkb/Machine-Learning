from docx import Document

doc = Document('大作业-细节.docx')
for para in doc.paragraphs:
    if para.text.strip():
        print(para.text)
