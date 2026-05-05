import os

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer



model_id = "BAAI/bge-m3"
save_dir = "./bge-m3-onnx"

# 1. 모델 및 토크나이저 로드 (export=True 설정으로 자동 ONNX 변환)
print(f"Exporting {model_id} to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(model_id, fix_mistral_regex=True, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 로컬 저장
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model successfully exported to {save_dir}")


# from optimum.onnxruntime import ORTQuantizer
# from optimum.onnxruntime.configuration import AutoQuantizationConfig

# # 위에서 저장한 ONNX 모델 경로
# onnx_model_path = "./bge-m3-onnx"
# quantized_model_path = "./bge-m3-onnx-int8"

# # 양자화 도구 초기화
# quantizer = ORTQuantizer.from_pretrained(onnx_model_path)

# # CPU 환경(AVX-512/VNNI)에 최적화된 동적 양자화 설정
# dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# # 양자화 수행
# quantizer.quantize(
#     save_dir=quantized_model_path,
#     quantization_config=dqconfig,
# )

# print(f"Quantized model saved to {quantized_model_path}")