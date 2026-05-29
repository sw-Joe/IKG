import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# 양자화 모델 경로 또는 원본 ONNX 경로
model_path = "./bge-m3-onnx-int8" 

model = ORTModelForFeatureExtraction.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# sentences = ["IKG 시스템 프로토타입 구현을 시작합니다.", "검색 엔진 최적화 테스트."]
sentences = ["IKG 시스템 프로토타입 구현을 시작합니다.", "프로토타입, 목업, 와이어프레임"]

# 토크나이징 및 추론
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="np")
outputs = model(**inputs)

# Dense Embedding 추출 (보통 마지막 레이어의 첫 번째 토큰인 [CLS] 사용)
# BGE-M3는 Normalization이 중요합니다.
embeddings = outputs.last_hidden_state[:, 0, :]
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print(f"Embedding Shape: {embeddings.shape}")
print(f"Cosine Similarity: {np.dot(embeddings[0], embeddings[1]):.4f}")