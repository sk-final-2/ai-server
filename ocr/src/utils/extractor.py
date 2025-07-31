from PyPDF2 import PdfReader
from docx import Document

# PDF는 PyPDF2를 사용해서 각 페이지의 텍스트를 이어붙입니다.
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# TXT 파일은 그냥 UTF-8로 읽기만 하면 됩니다.
def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# DOCX는 python-docx로 문단별 텍스트를 추출합니다.
def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])