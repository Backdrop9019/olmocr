# OlmOCR → Qwen3-VL 변환 작업 일지

## 목표
OlmOCR의 데이터 파이프라인을 유지하면서 Qwen2.5-VL → Qwen3-VL로 전환하여 학습

---

## 완료된 작업

### 1. 기본 설정 및 구조 변경
- ✅ Qwen3-VL imports 수정 (`Qwen3VLForConditionalGeneration` 등)
- ✅ Processor 설정: `min_pixels/max_pixels` + `size` dict 동시 업데이트 필요
- ✅ `max_image_size` 제거 - 불필요한 중간 resize 로직 삭제
- ✅ 이미지 파이프라인 단순화: PDFRenderer → Qwen Processor (2단계)

### 2. DeepSpeed 설정
- ✅ YAML indentation 수정 (`deepspeed:` config가 `training:` 섹션 안에 있어야 함)
- ✅ Qwen의 공식 `zero2.json` 사용 (CPU offload 제거, H100 80GB에 불필요)
- ✅ 위치: `/home/kyungho/frameworks/olmocr/olmocr/train/configs/qwen3/deepspeed_zero2.json`

### 3. 빈 배치 처리 문제 해결 ⭐ 핵심
**문제:** OlmOCR은 커스텀 training loop 사용, 우리는 HF Trainer 사용
- OlmOCR: `if batch is None: continue` (한 곳에서만 처리)
- Qwen3: HF Trainer가 내부적으로 여러 곳에서 배치 접근 → None 처리 안됨

**해결:** `QwenTrainer` 클래스 생성 (train_qwen3.py:136-190)
```python
class QwenTrainer(Trainer):
    # 1. FLOPs 계산 시 None 처리
    def floating_point_ops(self, inputs):
        if inputs is None or not inputs:
            return 0
        return super().floating_point_ops(inputs)

    # 2. 배치 크기 확인 시 None 처리 (gradient accumulation 대응)
    def _get_num_items_in_batch(self, batch, device=None):
        if batch is None or not batch:
            return 0
        if isinstance(batch, (list, tuple)) and len(batch) > 0 and batch[0] is None:
            return 0
        return super()._get_num_items_in_batch(batch, device)

    # 3. Training 시 None 처리
    def training_step(self, model, inputs, num_items_in_batch=None):
        if inputs is None or not inputs or len(inputs) == 0:
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        return super().training_step(model, inputs, num_items_in_batch)

    # 4. Evaluation 시 None 처리 (GPU 텐서 필수!)
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if inputs is None or not inputs or len(inputs) == 0:
            return (torch.tensor(0.0, device=model.device), None, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
```

**Collator 변경:**
- `return {}` → `return None` (OlmOCR 방식)
- 빈 배치일 때 None 리턴, QwenTrainer가 조용히 스킵

### 4. 로깅 정리
- ✅ Collator의 verbose logging 비활성화 (`verbose = False`)
- ✅ 데이터 로딩이 잘 되므로 상세 로그 불필요

### 5. Weight Tying 확인
- ✅ `lm_head.weight` missing 경고는 정상 (weight tying 사용 중)
- ✅ `tie_word_embeddings: True` → lm_head가 embed_tokens와 weight 공유

---

## 주요 파일 위치

### 학습 스크립트
- **메인 학습 코드**: `/home/kyungho/frameworks/olmocr/olmocr/train/train_qwen3.py`
- **데이터 어댑터**: `/home/kyungho/frameworks/olmocr/olmocr/train/qwen_data_adapter.py`

### 설정 파일
- **8B 프로덕션 config**: `/home/kyungho/frameworks/olmocr/olmocr/train/configs/qwen3/qwen3_8b_olmocr.yaml`
- **2B 디버그 config**: `/home/kyungho/frameworks/olmocr/olmocr/train/configs/qwen3/qwen3_2b_debug.yaml`
- **DeepSpeed config**: `/home/kyungho/frameworks/olmocr/olmocr/train/configs/qwen3/deepspeed_zero2.json`

### 실행 스크립트
- **학습 시작**: `olmocr/train/configs/qwen3/run_qwen3_8b.sh`
- **로그 보기**: `olmocr/train/configs/qwen3/view_logs.sh`
- **학습 중지**: `olmocr/train/configs/qwen3/stop_training.sh`

---

## 현재 설정 요약 (qwen3_8b_olmocr.yaml)

### 모델
- Qwen3-VL-8B-Instruct
- bf16, flash_attention_2
- tune_mm_llm + tune_mm_mlp (vision frozen)

### 데이터
- 4개 train datasets (books, documents, loc, national_archives)
- 4개 eval datasets
- 이미지 해상도: 1288 → 1280 (smart_resize)
- `min_pixels: 12544` (28×28×16)
- `max_pixels: 1646400` (28×28×2100)

### 학습
- Batch size: 1 per device
- Gradient accumulation: 16 → effective batch 64 (4 GPU)
- Max length: 8192 tokens
- Learning rate: 2e-5, linear schedule, warmup 0.1
- 1 epoch
- DeepSpeed ZeRO-2
- Gradient checkpointing enabled
- Torch compile enabled

### 체크포인트
- Output: `/home/kyungho/olmocr-qwen3-8b/`
- Save/eval every 500 steps
- Logging every 10 steps

---

## 중요 개념 및 교훈

### 1. Processor 설정의 함정
Qwen3-VL processor는 `min_pixels/max_pixels` **attributes**와 `size` **dict** 둘 다 업데이트해야 함:
```python
processor.image_processor.min_pixels = value  # 이것만으로는 부족!
processor.image_processor.max_pixels = value
processor.image_processor.size['shortest_edge'] = min_pixels  # 이것도 필요!
processor.image_processor.size['longest_edge'] = max_pixels
```
→ `smart_resize`가 실제로 읽는 건 `size` dict!

### 2. OlmOCR vs HF Trainer 아키텍처 차이
- **OlmOCR**: 커스텀 루프 → 한 곳에서만 None 체크
- **HF Trainer**: 내부에서 배치를 여러 함수가 건드림 → 여러 곳에서 None 처리 필요
- **교훈**: 처음부터 비교 분석했어야 했음

### 3. 빈 배치 처리 패턴
- Collator에서 `None` 리턴
- Trainer에서 다음 메서드들 오버라이드:
  1. `floating_point_ops` - FLOPs 계산
  2. `_get_num_items_in_batch` - 배치 크기 확인 (gradient accumulation 고려!)
  3. `training_step` - 학습
  4. `prediction_step` - 평가 (GPU 텐서 필수!)

### 4. Gradient Accumulation 주의사항
- `batch`가 `[None]` 형태로 올 수 있음 (리스트 안의 None)
- `if batch is None:` 만으로는 불충분
- `if batch[0] is None:` 체크도 필요

### 5. Multi-GPU Evaluation
- Dummy 텐서도 **반드시 GPU에 생성**해야 함
- `torch.tensor(0.0)` ❌ → CPU
- `torch.tensor(0.0, device=model.device)` ✅ → GPU

---

## 데이터 파이프라인

### Validation vs Runtime Filtering
**Validation (dataset 초기화 시):**
- PDF 파일 존재 여부만 체크
- 열 수 있는지만 확인

**Runtime Filtering (학습 중):**
- LaTeX 렌더링 실패 → `return None`
- 예: `\left[` 있는데 `\right]` 없으면 KaTeX 파싱 실패
- Collator에서 None 필터링, 전부 None이면 배치 스킵

### 파이프라인 단계 (basic_pipeline)
1. FrontMatterParser
2. FilterOutRotatedDocuments
3. ReformatLatexBoldItalic
4. DatasetTextRuleFilter
5. PDFRenderer (target_longest_image_dim: 1288)
6. RotationAugmentation (2% 확률)
7. NewYamlFinetuningPromptWithNoAnchoring
8. FrontMatterOutputFormat
9. InstructUserMessages
10. Tokenizer

---

## 다음 단계

### 즉시 실행 가능
```bash
cd /home/kyungho/frameworks/olmocr
./olmocr/train/configs/qwen3/run_qwen3_8b.sh
```

### 학습 모니터링
```bash
# 로그 실시간 확인
./olmocr/train/configs/qwen3/view_logs.sh

# 또는
tail -f /home/kyungho/olmocr-qwen3-8b/logs/train_*.log

# TensorBoard (wandb 대신)
tensorboard --logdir /home/kyungho/olmocr-qwen3-8b/
```

### 학습 중 확인사항
1. **첫 몇 step**: Torch compile 시간 (정상)
2. **Loss 감소**: 정상적으로 학습되는지
3. **Bad samples**: 가끔 LaTeX 에러 뜨는 건 정상 (스킵됨)
4. **GPU 메모리**: nvidia-smi로 OOM 없는지 확인
5. **Eval metrics**: 500 step마다 eval loss 체크

### 학습 후 체크리스트
- [ ] Checkpoint 저장 확인: `/home/kyungho/olmocr-qwen3-8b/checkpoint-*`
- [ ] 최종 모델 저장 확인
- [ ] Eval metrics 비교 (처음 vs 마지막)
- [ ] 샘플 inference 테스트

---

## 트러블슈팅

### DeepSpeed 안 켜질 때
- YAML indentation 확인 (`deepspeed:` 가 `training:` 섹션 안에 있는지)
- Config 파일 경로 확인

### OOM (Out of Memory)
- `per_device_train_batch_size` 줄이기 (1 → 1 유지, 이미 최소)
- `gradient_accumulation_steps` 늘리기 (16 → 32)
- `max_length` 줄이기 (8192 → 4096)

### 학습이 너무 느림
- `torch_compile` 첫 실행 시 컴파일 시간 김 (정상)
- `dataloader_num_workers` 조정 (4 → 8)
- `gradient_checkpointing` 끄기 (메모리 여유 있으면)

### Bad LaTeX equations
- 정상 동작: 자동으로 스킵됨
- 너무 많으면 데이터셋 문제 (전체의 <1% 정도는 OK)

---

## 참고 자료

### OlmOCR 원본
- 학습 코드: `/home/kyungho/frameworks/olmocr/olmocr/train/train.py`
- Collator: QwenDataCollator (line 79-131)
- 커스텀 training loop 사용

### Qwen3-VL 공식
- 경로: `/home/kyungho/frameworks/Qwen3-VL`
- DeepSpeed config: `qwen-vl-finetune/scripts/zero2.json`

### 환경
- Python: 3.11
- Conda env: `olmocr-qwen3`
- GPU: H100 80GB × 4

---

## 마지막 업데이트
2025-11-24 03:30 KST

**상태**: ✅ 학습 준비 완료, 8B 프로덕션 config 검증 완료
**다음**: `run_qwen3_8b.sh` 실행
