from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch

# 1. 모델/토크나이저 로드
model = ElectraForSequenceClassification.from_pretrained("koelectra")
tokenizer = ElectraTokenizer.from_pretrained("koelectra")

# 2. 예측하고 싶은 문장
question = "최신 AI 기술을 적용한 프로젝트에서 실제 문제를 해결하는 경험은 있는가?"
answer = "히히히히 똥!"

# 3. 전처리
inputs = tokenizer(
    question + " [SEP] " + answer,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=256,
)

# 4. 예측
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()

# 5. 결과 출력
labels = ["terminate", "continue"]
print(f"예측 결과: {labels[predicted_class]} ({probs[0][predicted_class]:.2f})")
