from docx import Document
import sys

paths = ['大作业-细节.docx', '单细胞转录组聚类分析大作业评分标准.docx']
for p in paths:
    try:
        doc = Document(p)
        print('\n' + '='*30)
        print('FILE:', p)
        print('='*30)
        for para in doc.paragraphs:
            if para.text.strip():
                print(para.text)
    except Exception as e:
        print('Error reading', p, e)
        sys.exit(1)
