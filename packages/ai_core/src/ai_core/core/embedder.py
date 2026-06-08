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
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_full_path}")

        # 1. ONNX Runtime 세션 초기화 (CPU 최적화 설정)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4 # CPU 코어 수에 맞게 스레드 할당
        self.session = ort.InferenceSession(model_full_path, sess_options, providers=['CPUExecutionProvider'])

        # 2. 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def encode(self, text: str) -> np.ndarray:
            """기존 단건 추론 로직 (기존 레거시 및 be_api/tasks.py 하위 호환 보존)"""
            inputs = self.tokenizer(
                text, padding=True, truncation=True, max_length=8192, return_tensors="np"
            )
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
                
            outputs = self.session.run(None, onnx_inputs)
            return outputs[0]


    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        [OPTIMIZED TRUE BATCH] 다량의 원문 컨텍스트 배열을 단 1회의 
        ONNX 텐서 연산으로 병렬 가속하여 고차원 밀도 행렬을 도출합니다.
        """
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        # 1. 100건 전체에 대한 병렬 토크나이징 (가장 긴 장문 문서를 기점으로 일괄 패딩 정렬)
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=8192, 
            return_tensors="np"
        )
        
        # 2. C++ ONNX Runtime 입력 매트릭스 변환
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        if "token_type_ids" in inputs:
            onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            
        # 3. 단 1회의 실행으로 전체 배치 차원 밀어붙이기집행
        outputs = self.session.run(None, onnx_inputs)
        
        # BGE-M3 Dense 임베딩의 특징인 CLS 토큰 풀링 결과 레이어([:, 0, :])를 취해 슬라이싱 추출 후 정규화
        dense_embeddings = outputs[0][:, 0, :]
        
        # 4. 수학적 상호 코사인 공간 보존을 위한 고속 정규화(L2 Normalization) 집행
        norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True) + 1e-9
        normalized_embeddings = dense_embeddings / norms
        
        return normalized_embeddings.astype(np.float32)

    
    def encode_sparse(self, text: str) -> dict:
        """
        입력 텍스트를 BGE-M3 Native Sparse Embedding으로 변환하여
        {token_id: weight} 구조의 가중치 딕셔너리를 반환합니다.
        
        ONNX 모델의 출력 중 Sparse 텐서(예: 'sparse_output' 또는 세 번째 출력 텐서)를 파싱합니다.
        """
        if not text.strip():
            return {}

        # 1. 텍스트 토크나이징 및 ONNX 입력 텐서 형태로 변환
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=8192,  # BGE-M3 최대 컨텍스트 길이 지원
            return_tensors="np"
        )
        
        # ONNX Runtime 입력 규격에 맞게 64비트 정수형 정렬
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        try:
            # 2. ONNX 모델 추론 수행
            outputs = self.session.run(None, onnx_inputs)
            
            # 3. Sparse Output 텐서 특정 및 추출
            # 일반적인 BGE-M3 ONNX 변환 스펙 상 [Dense(0), ColBERT(1), Sparse(2)] 순서로 배치됩니다.
            if len(outputs) < 3:
                # 모델 출력 스펙이 Dense 전용으로 빌드되었을 시 예외 처리 및 빈 가중치 반환
                return {}
                
            sparse_output = outputs[2]  # Shape: (batch_size, sequence_length, 1) 또는 (batch_size, dict_size)
            input_ids = inputs["input_ids"][0]
            
            # 4. 토큰 가중치 딕셔너리 매핑 및 차원 예외 처리 (Edge Case Guard)
            token_weights = {}
            
            # Case A: 출력이 시퀀스 길이별 가중치 선형 결합 형태인 경우 (일반적인 HuggingFace 오프라인 내보내기 스펙)
            if len(sparse_output.shape) == 3:
                weights = sparse_output[0]  # 첫 번째 배치의 (seq_len, 1) 추출
                for idx, token_id in enumerate(input_ids):
                    # Special Token (CLS, SEP, PAD) 필터링 및 유효 가중치 적재
                    if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                        continue
                    
                    w_val = float(weights[idx][0])
                    if w_val > 0.0:  # 0 이하의 노이즈 가중치 제거를 통한 데이터 압축
                        token_id_int = int(token_id)
                        # 중복 단어 토큰 출현 시 최대 가중치 유지(Max-pooling) 전략 적용
                        token_weights[token_id_int] = max(token_weights.get(token_id_int, 0.0), w_val)
            
            # Case B: 출력이 이미 어휘집 차원(Vocabulary Size) 전체로 Relu 연산이 끝난 고정 차원 벡터 형태인 경우
            elif len(sparse_output.shape) == 2:
                vocab_weights = sparse_output[0]
                # 임계값 이상 활성화된 토큰만 필터링하여 스파스성(Sparsity) 유지
                nonzero_indices = np.nonzero(vocab_weights)[0]
                for idx in nonzero_indices:
                    w_val = float(vocab_weights[idx])
                    if w_val > 0.0:
                        token_weights[int(idx)] = w_val

            return token_weights

        except Exception as e:
            # 실시간 검색 파이프라인 무너짐 방지를 위한 무장애(Graceful Degradation) 방어선 구축
            print(f"[WARNING] Native Sparse 인코딩 중 예외 발생: {e}")
            return {}