# Training

이 폴더는 **확률분포 시 생성기**의 LoRA 파인튜닝 작업을 위한 공간이다.

## 프로젝트 고정 기준

- 기본 모델: `Qwen/Qwen2.5-0.5B-Instruct`
- 학습 방식: Colab에서 LoRA 또는 QLoRA 파인튜닝
- 입력: 사용자 경험 프롬프트
- 출력: 무조건 3줄 한국어 시
- 슬라이더: `언어 도약도` 하나만 사용
- 데이터셋: 저작권 없는 고전시 기반 200개 + 현대 일상 경험 50~100개
- 산출물: LoRA adapter와 merged model을 모두 Hugging Face Hub에 업로드
- 금지: 하드코딩 시 템플릿, demo fallback, validation 실패 출력 숨김

## 파일 구성

```text
training/
├── train_lora_colab.ipynb
├── prepare_dataset.py
├── merge_lora.py
└── README.md
```

## 데이터 준비

원천 데이터는 아래 파일에 넣는다.

```text
data/classic_poems_raw.jsonl
data/experience_poem_pairs.jsonl
```

각 줄은 JSON object 하나이며 최소 필드는 다음과 같다.

```json
{"experience":"비 오는 밤 정류장에서 버스를 기다림","poem":"정류장은 젖은 불빛을 들고\n버스는 아직 오지 않은 문장처럼 멀다\n나는 비의 안쪽에서 조금 늦어진다"}
```

학습용 파일 생성:

```bash
python training/prepare_dataset.py
```

출력:

```text
data/train.jsonl
```

## LoRA 학습 흐름

1. `data/train.jsonl` 생성
2. Colab에서 `training/train_lora_colab.ipynb` 실행
3. LoRA adapter 저장
4. adapter를 Hugging Face Hub에 업로드
5. `training/merge_lora.py`로 merged model 생성
6. merged model을 Hugging Face Hub에 업로드
7. Hugging Face Space backend에서 merged model을 우선 로드

## merged model 생성 예시

```bash
python training/merge_lora.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter z-unghyun/poem-generator-lora-adapter \
  --output_dir outputs/poem-generator-merged \
  --push_to_hub \
  --hub_model_id z-unghyun/poem-generator-merged
```

## validation 원칙

검증은 출력 차단 장치가 아니라 진단 장치다.

- 성공: 원문 출력 + `validation_status=valid`
- 실패: 원문 그대로 출력 + `validation_status=invalid_shown`
- 실패 사유: `validation_reason`에만 표시
