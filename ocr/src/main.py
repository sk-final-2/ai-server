from fastapi import FastAPI, UploadFile, File
from utils.extractor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_docx
)
import tempfile, os, re
from utils.text_cleaner import clean_spacing

app = FastAPI()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # 공백 정리
    return text.strip()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    _, ext = os.path.splitext(filename)

    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 확장자별 텍스트 추출
        if ext == ".pdf":
            raw_text = extract_text_from_pdf(tmp_path)
        elif ext == ".txt":
            raw_text = extract_text_from_txt(tmp_path)
        elif ext == ".docx":
            raw_text = extract_text_from_docx(tmp_path)
        else:
            return JSONResponse(status_code=400, content={"error": f"{ext} 형식은 지원하지 않습니다."})

        cleaned_text = clean_spacing(raw_text)
        return {"ocr_output": cleaned_text}

    finally:
        # 임시 파일 정리
        os.remove(tmp_path)