import re

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
