from typing import List, Dict, Any 
import fitz # PyMuPDF 
import regex as re
from docx import Document
from src.utils.text_cleaner import clean_pdf_text

# 문자열의 간격 조정
def _avg_char_width(spans: List[Dict[str, Any]]) -> float:
    widths = []
    for sp in spans:
        x0, y0, x1, y1 = sp["bbox"]
        w = max(x1 - x0, 1.0)
        n = max(len(sp.get("text", "")), 1)
        widths.append(w / n)
    return (sum(widths) / len(widths)) if widths else 4.0

def extract_text_from_pdf_pymupdf(path: str) -> str:
    """
    블록/라인/스팬 좌표 기반 재구성:
    - page.get_text("dict") → blocks → lines → spans
    - 같은 라인에서 스팬 간 가로 간격이 평균 글자폭 * gap_factor 보다 크면 공백 삽입
    - 블록/라인은 y→x 순으로 정렬
    """
    doc = fitz.open(path)
    pages_out: List[str] = []
    gap_factor = 0.55   # 공백 삽입 민감도

    for page in doc:
        d = page.get_text("dict")
        blocks = [b for b in d.get("blocks", []) if b.get("type", 0) == 0]

        # 블록을 y(상단) → x(좌측)으로 정렬
        blocks.sort(key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))

        lines_out: List[str] = []
        for b in blocks:
            # 라인 정렬
            b_lines = b.get("lines", [])
            b_lines.sort(key=lambda ln: (round(ln["bbox"][1], 1), round(ln["bbox"][0], 1)))

            for ln in b_lines:
                spans = ln.get("spans", [])
                if not spans:
                    continue
                # 스팬을 x0 기준으로 정렬
                spans.sort(key=lambda sp: sp["bbox"][0])

                # 평균 글자 폭 추정
                avg_cw = _avg_char_width(spans)

                pieces = []
                prev_x1 = None
                for sp in spans:
                    text = sp.get("text", "")
                    x0, y0, x1, y1 = sp["bbox"]

                    if prev_x1 is not None:
                        gap = x0 - prev_x1
                        if gap > avg_cw * gap_factor:
                            pieces.append(" ")

                    # 스팬 내부에 원래 공백이 있으면 그대로 살려짐
                    pieces.append(text)
                    prev_x1 = x1

                line_text = ''.join(pieces).strip()
                if line_text:
                    lines_out.append(line_text)

        page_text = '\n'.join(lines_out)
        pages_out.append(clean_pdf_text(page_text))

    return '\n\n'.join(pages_out)

# TXT 파일은 그냥 UTF-8로 읽기만 하면 됩니다.
def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# DOCX는 python-docx로 문단별 텍스트를 추출합니다.
def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])