import regex as re
import unicodedata

def clean_spacing(text: str) -> str:
    # 숫자 + 단위(년/월/일/시/분/초 등) 사이 공백 제거
    text = re.sub(r'(\d+)\s+(년|월|일|시|분|초|%)', r'\1\2', text)

    # 단위 뒤에 붙는 조사 앞 공백 제거 (예: "4월 ~ 6월" → "4월~6월")
    text = re.sub(r'\s*([~\-–])\s*', r'\1', text)

    # 숫자 + 단위(년, 월, %, 건, 명, 회 등) 사이 공백 제거
    text = re.sub(r'(\d+)\s+(년|월|일|시|분|초|%|건|명|회|점|개|위|등)', r'\1\2', text)

    # 조사 사이 공백 제거
    text = re.sub(r'([가-힣a-zA-Z0-9])\s+([은는이가의을를도과와에에서부터까지])', r'\1\2', text)
    
    # 문장 부호 앞뒤 공백 제거
    text = re.sub(r'\s+([.,!?;:\'\"”’)\]])', r'\1', text)
    text = re.sub(r'([(\[“‘])\s+', r'\1', text)

    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def clean_pdf_text(text: str) -> str:
    """
    PDF 추출용 텍스트 후처리:
    - 제어문자 제거
    - 하이픈 줄바꿈 교정
    - 불필요한 공백/개행 정리
    - NFKC 정규화
    """
    # 제어문자 제거
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', text)
    text = re.sub(r'\p{Cc}|\p{Cf}', ' ', text)

    # 하이픈 + 줄바꿈 교정
    text = re.sub(r'-\s*\n', '', text)

    # 불필요한 공백/개행 정리
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 한글/영문·숫자 경계 공백 삽입
    text = re.sub(r'([\p{IsHangul}])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9])([\p{IsHangul}])', r'\1 \2', text)

    # NFKC 정규화
    text = unicodedata.normalize('NFKC', text)

    # 최종 공백 정리
    return text.strip()