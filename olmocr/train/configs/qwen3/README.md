# Qwen3-VL Training Configurations

OlmOCR v0.4.0 레시피 기반 Qwen3-VL 학습 설정 파일

## 📁 파일 구조

```
configs/qwen3/
├── qwen3_2b_debug.yaml      # 2B 디버깅용 (빠른 실행)
├── qwen3_2b.yaml             # 2B 프로덕션
├── qwen3_4b.yaml             # 4B 프로덕션
├── qwen3_8b.yaml             # 8B 프로덕션
├── deepspeed_zero2.json      # DeepSpeed ZeRO-2
├── deepspeed_zero3.json      # DeepSpeed ZeRO-3
└── README.md                 # 이 파일
```

## 🎯 설정 파일 선택 가이드

### 1. **qwen3_2b_debug.yaml** - 빠른 디버깅용

**용도**: 코드 테스트, 파이프라인 검증

**특징**:
- 모델: Qwen3-VL-2B-Instruct
- max_steps: 100 (빠른 종료)
- 이미지: 1024px (낮은 해상도)
- 배치: 2 x 4 = 8 (작은 배치)
- 시퀀스: 2048 (짧음)
- torch_compile: False (빠른 시작)
- eval_steps: 20 (자주 체크)

**실행 시간**: ~30분

```bash
python -m olmocr.train.train_qwen3 \
  --olmocr_config_path configs/qwen3/qwen3_2b_debug.yaml
```

---

### 2. **qwen3_2b.yaml** - 2B 프로덕션

**용도**: 실제 학습 (작은 모델)

**특징**:
- 모델: Qwen3-VL-2B-Instruct
- OlmOCR v0.4.0 설정 그대로
- 이미지: 1288px
- 배치: 1 x 32 = 32
- 시퀀스: 8192
- torch_compile: True
- 1 epoch 학습

**필요 GPU**: 1x 24GB (A6000, RTX 4090 등)

```bash
python -m olmocr.train.train_qwen3 \
  --olmocr_config_path configs/qwen3/qwen3_2b.yaml
```

---

### 3. **qwen3_4b.yaml** - 4B 프로덕션

**용도**: 실제 학습 (중간 크기)

**특징**:
- 모델: Qwen3-VL-4B-Instruct
- OlmOCR v0.4.0 설정 그대로
- 배치: 1 x 32 = 32
- **DeepSpeed ZeRO-2 사용**

**필요 GPU**: 2-4x 24GB (A6000 등)

```bash
torchrun --nproc_per_node=4 \
  olmocr/train/train_qwen3.py \
  --olmocr_config_path configs/qwen3/qwen3_4b.yaml \
  --deepspeed configs/qwen3/deepspeed_zero2.json
```

---

### 4. **qwen3_8b.yaml** - 8B 프로덕션

**용도**: 실제 학습 (최고 품질)

**특징**:
- 모델: Qwen3-VL-8B-Instruct
- OlmOCR v0.4.0 설정 그대로
- 배치: 1 x 32 = 32
- ZeRO-2 또는 ZeRO-3 사용 가능

**필요 GPU**:
- 단일: 1x 80GB (A100, H100)
- 멀티: 4-8x 24GB (DeepSpeed ZeRO-2/3)

```bash
# 단일 GPU (80GB)
python -m olmocr.train.train_qwen3 \
  --olmocr_config_path configs/qwen3/qwen3_8b.yaml

# 멀티 GPU (ZeRO-2)
torchrun --nproc_per_node=8 \
  olmocr/train/train_qwen3.py \
  --olmocr_config_path configs/qwen3/qwen3_8b.yaml \
  --deepspeed configs/qwen3/deepspeed_zero2.json

# 멀티 GPU (ZeRO-3, 최대 메모리 절약)
torchrun --nproc_per_node=8 \
  olmocr/train/train_qwen3.py \
  --olmocr_config_path configs/qwen3/qwen3_8b.yaml \
  --deepspeed configs/qwen3/deepspeed_zero3.json
```

---

## 🔧 DeepSpeed 설정

### ZeRO-2 vs ZeRO-3

| Stage | 메모리 절약 | 속도 | MoE 지원 | 사용 권장 |
|-------|-----------|------|---------|---------|
| **ZeRO-2** | 중간 (~50%) | 빠름 | ✅ | 일반적 |
| **ZeRO-3** | 최대 (~70%) | 느림 | ❌ | 메모리 부족시 |

**선택 기준**:
- **ZeRO-2**: 일반적인 경우, MoE 모델
- **ZeRO-3**: GPU 메모리 부족시, Dense 모델만

---

## 📊 OlmOCR v0.4.0 핵심 설정

모든 설정은 다음 레시피 기반:
`qwen25_vl_olmocrv4_rotation_1epoch_mix_1025_filtered.yaml`

### 공통 설정

```yaml
# 학습
num_train_epochs: 1
learning_rate: 2e-5
warmup_ratio: 0.1
gradient_accumulation_steps: 32
gradient_checkpointing: false  # OlmOCR v0.4.0은 끔
torch_compile: true            # OlmOCR v0.4.0 사용

# 이미지
target_longest_image_dim: 1288
max_pixels: 1653248  # 1288*1288
min_pixels: 784      # 28*28

# 파이프라인
RotationAugmentation: 0.02     # 2% 확률
FilterOutRotatedDocuments: true
DatasetTextRuleFilter: true

# 체크포인트
eval_steps: 500
save_steps: 500
save_total_limit: 5
```

### 디버그 설정 차이점

```yaml
# qwen3_2b_debug.yaml만 다른 부분
max_steps: 100                 # 빠른 종료
target_longest_image_dim: 1024 # 낮은 해상도
model_max_length: 2048         # 짧은 시퀀스
per_device_train_batch_size: 2 # 작은 배치
gradient_accumulation_steps: 4
torch_compile: false           # 빠른 시작
eval_steps: 20                 # 자주 체크
```

---

## ⚠️ 주의사항

### 1. 데이터 경로
모든 설정의 `root_dir`을 실제 경로로 변경:
```yaml
root_dir: /data/olmOCR-mix-1025/processed_01_books_train_iabooks/
```

### 2. 출력 경로
`output_dir`을 원하는 경로로 변경:
```yaml
output_dir: /home/kyungho/olmocr-qwen3-7b/
```

### 3. wandb 로깅
wandb 사용시 로그인 필요:
```bash
wandb login
```

### 4. Flash Attention
설치 필수:
```bash
pip install flash-attn>=2.7.4 --no-build-isolation
```

### 5. DeepSpeed 설정
- ZeRO-2: 일반적인 멀티 GPU 학습
- ZeRO-3: 메모리 부족시 사용 (더 느림)

---

## 🚀 빠른 시작

### 1. 디버깅 (5분 테스트)
```bash
python -m olmocr.train.train_qwen3 \
  --olmocr_config_path configs/qwen3/qwen3_2b_debug.yaml \
  --max_steps 10
```

### 2. 실제 학습
```bash
# 2B (단일 GPU)
python -m olmocr.train.train_qwen3 \
  --olmocr_config_path configs/qwen3/qwen3_2b.yaml

# 8B (멀티 GPU)
torchrun --nproc_per_node=8 \
  olmocr/train/train_qwen3.py \
  --olmocr_config_path configs/qwen3/qwen3_8b.yaml \
  --deepspeed configs/qwen3/deepspeed_zero2.json
```

---

## 🤔 애매한 부분 (결정 필요)

아래 값들은 OlmOCR v0.4.0과 동일하게 설정했으나, 조정 가능:

### ✅ 확정된 것 (OlmOCR v0.4.0 그대로)
- `num_train_epochs: 1`
- `learning_rate: 2e-5`
- `warmup_ratio: 0.1`
- `gradient_accumulation_steps: 32`
- `eval_steps: 500`
- `save_steps: 500`
- `rotation_probability: 0.02`
- `image_size: 1288`

### 🤷 선택 가능한 것

1. **gradient_checkpointing**:
   - OlmOCR v0.4.0: `false` (빠름, 메모리 많이 사용)
   - 메모리 부족시: `true` (느림, 메모리 절약)

2. **torch_compile**:
   - OlmOCR v0.4.0: `true` (첫 실행 느림, 이후 빠름)
   - 디버깅시: `false` (바로 시작)

3. **deepspeed stage**:
   - 일반: ZeRO-2 (빠름)
   - 메모리 부족: ZeRO-3 (느림, Dense만)

4. **wandb vs tensorboard**:
   - 기본: wandb (온라인)
   - 오프라인: tensorboard (로컬)

---

## 📞 문제 해결

### OOM (메모리 부족)
```yaml
# 1. 작은 배치
per_device_train_batch_size: 1
gradient_accumulation_steps: 64

# 2. Checkpointing 활성화
gradient_checkpointing: true

# 3. ZeRO-3 사용
deepspeed: configs/qwen3/deepspeed_zero3.json

# 4. 작은 시퀀스
model_max_length: 4096
```

### 느린 학습
```yaml
# 1. Checkpointing 끄기
gradient_checkpointing: false

# 2. Compile 활성화
torch_compile: true

# 3. ZeRO-2 사용
deepspeed: configs/qwen3/deepspeed_zero2.json
```

### 디버깅
```bash
# Debug 설정 + 매우 짧게
python -m olmocr.train.train_qwen3 \
  --olmocr_config_path configs/qwen3/qwen3_2b_debug.yaml \
  --max_steps 5 \
  --logging_steps 1
```