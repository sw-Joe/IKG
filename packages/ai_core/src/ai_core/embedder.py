import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class BGEEmbedder:
    def __init__(self, model_path: str, file_name: str = "model.onnx"):
        """
        BGE-M3 ONNX 모델을 로드하여 추론 엔진을 초기화합니다.
        """
        model_full_path = os.path.join(model_path, file_name)
        if not os.path.exists(model_full_path):
            print(f"[WARN] ONNX 모델 파일을 찾을 수 없습니다 ({model_full_path}). Mock DUMMY 모드로 동작합니다.")
            self.dummy_mode = True
            return

        self.dummy_mode = False
        # 1. ONNX Runtime 세션 초기화 (CPU 최적화 설정)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4 # CPU 코어 수에 맞게 스레드 할당
        self.session = ort.InferenceSession(model_full_path, sess_options, providers=['CPUExecutionProvider'])

        # 2. 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def encode(self, text: str) -> np.ndarray:
        """
        입력 텍스트를 1024차원의 정규화된 벡터로 변환합니다.
        """
        if self.dummy_mode:
            # 해시 함수 기반으로 텍스트별 고유하고 결정론적인 모의 벡터 생성 (L2 정규화 준수)
            import hashlib
            h = hashlib.sha256(text.encode('utf-8')).digest()
            np.random.seed(int.from_bytes(h[:4], byteorder='big'))
            vec = np.random.randn(1, 1024).astype('float32')
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            return vec / (norm + 1e-9)

        # 3. 텍스트 전처리 (Max length 8192 토큰 대응)
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=8192, 
            return_tensors="np"
        )
        
        # ONNX 모델의 입력 형식에 맞게 변환 (int64)
        onnx_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

        # 4. 모델 추론 실행
        outputs = self.session.run(None, onnx_inputs)
        
        # 5. Dense Embedding 추출 (첫 번째 레이어의 CLS 토큰 사용)
        # BGE-M3 ONNX 출력 구조에 따라 인덱싱이 달라질 수 있으나, 일반적으로 0번째 출력의 [:, 0, :]입니다.
        last_hidden_state = outputs[0]
        embeddings = last_hidden_state[:, 0, :]

        # 6. L2 정규화 (유사도 계산 효율 극대화)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norm + 1e-9)
        
        return normalized_embeddings.astype('float32')