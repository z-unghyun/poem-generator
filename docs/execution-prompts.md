# 확률분포 시 생성기 실행 프롬프트 체크리스트

이 문서는 `확률분포 시 생성기` 프로젝트를 ChatGPT, Colab, GitHub, Hugging Face Space, Vercel을 연결해 단계별로 진행하기 위한 실행용 프롬프트 모음이다.

각 단계는 독립적으로 실행할 수 있도록 작성되어 있다. 예를 들어 `4단계만 진행해줘`, `10단계만 진행해줘`처럼 지시할 수 있다.

---

## 0. 고정 프로젝트 기준

```text
너는 지금부터 내 개인 프로젝트 “확률분포 시 생성기”의 개발 보조자야.

프로젝트 기준은 다음과 같아.

- 프롬프트 모드는 삭제한다.
- 최종 앱은 파인튜닝된 모델 기반 단일 모드만 가진다.
- 입력은 사용자 경험 프롬프트다.
- 출력은 무조건 3줄 시다.
- 슬라이더는 “언어 도약도” 하나만 남긴다.
- 데이터셋은 저작권 없는 고전시 200개 + 현대 일상 경험 50~100개로 만든다.
- 기본 모델은 Qwen/Qwen2.5-0.5B-Instruct다.
- Colab에서 LoRA 파인튜닝한다.
- adapter와 merged model을 둘 다 Hugging Face Hub에 업로드한다.
- Hugging Face Space는 무료 CPU Space에서도 돌아가는 것을 우선한다.
- Vercel frontend → Vercel API proxy → Hugging Face Space backend 구조를 사용한다.
- validation은 출력을 숨기지 않는다.
- 검증 실패해도 원문을 그대로 보여주고 validation_status, validation_reason만 표시한다.
- 하드코딩 시 템플릿, demo fallback, 실패 출력 숨김은 금지한다.

이 기준을 이후 모든 답변에 적용해줘.
```

---

## 전체 진행 순서

```text
0. 프로젝트 기준 고정
1. GitHub 구조 정리
2. 데이터셋 스키마 확정
3. 고전시 200편 소스 선정
4. 고전시 → 3줄 시 → 경험 프롬프트 데이터 제작
5. 현대 일상 경험 데이터 제작
6. 데이터셋 검수 스크립트 제작
7. Colab LoRA 학습 노트북 제작
8. Colab에서 학습 실행
9. adapter + merged model Hugging Face Hub 업로드
10. Hugging Face Space backend 교체
11. Vercel frontend/API 단순화
12. 전체 연결 테스트
13. 생성 효율 개선
14. 실험 로그/결과 정리
15. 데이터셋 개선 루프
16. README 작성
17. 최종 리팩터링
18. 운영 체크리스트 정리
```

---

# 1단계. GitHub 구조 정리

## 목적

GitHub repo를 학습, 데이터, 백엔드, 프론트엔드, 문서 구조로 정리한다.

## 실행 프롬프트

```text
내 GitHub repo는 z-unghyun/poem-generator야.

다음 구조로 프로젝트를 정리하고 싶어.

poem-generator/
├── index.html
├── api/
│   └── generate-poem.js
├── hf-space-backend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── training/
│   ├── train_lora_colab.ipynb
│   ├── prepare_dataset.py
│   ├── merge_lora.py
│   └── README.md
├── data/
│   ├── classic_poems_raw.jsonl
│   ├── experience_poem_pairs.jsonl
│   └── train.jsonl
└── docs/
    ├── project-plan.md
    └── execution-prompts.md

지금 repo 상태를 확인하고, 필요한 폴더와 빈 파일 또는 초안 파일을 생성해줘.
단, 기존 파일을 함부로 삭제하지 말고 변경 전 어떤 파일을 만들거나 수정할지 먼저 요약해줘.
```

## 완료 기준

```text
training/ 폴더 생성
data/ 폴더 생성
docs/ 폴더 생성
기존 index.html, api/generate-poem.js, hf-space-backend/app.py 보존
```

---

# 2단계. 데이터셋 스키마 확정

## 목적

고전시 기반 데이터와 현대 일상 경험 데이터를 같은 JSONL 구조로 관리할 수 있게 스키마를 확정한다.

## 실행 프롬프트

```text
확률분포 시 생성기용 학습 데이터셋 스키마를 확정해줘.

목표:
- 입력: 경험 프롬프트
- 출력: 3줄 한국어 시
- 고전시 기반 데이터와 현대 일상 경험 데이터를 같은 포맷으로 합칠 수 있어야 함
- JSONL로 저장할 예정
- Colab에서 datasets 라이브러리로 쉽게 불러올 수 있어야 함

원하는 기본 필드:
- id
- source_type: classic_poem 또는 modern_daily
- source_title
- source_author
- source_text
- experience
- poem
- style_tags

출력:
1. 최종 JSONL 스키마
2. 예시 데이터 5개
3. 필수 필드와 선택 필드 구분
4. 데이터 검수 기준
5. train.jsonl로 변환할 때 사용할 최소 포맷
```

## 완료 기준

```text
마스터 데이터셋 스키마 확정
학습용 train.jsonl 최소 스키마 확정
poem은 정확히 3줄 기준 확정
```

---

# 3단계. 고전시 200편 후보 선정

## 목적

저작권 문제가 낮고 이미지성이 강한 고전시 후보 200편을 선정한다.

## 실행 프롬프트 3-1: 1~50번

```text
저작권 문제가 없는 한국 고전시/고전 시가/근대 이전 시 작품 200편 후보를 선정하려고 해.

조건:
- 저작권 문제가 낮은 고전 작품 중심
- 현대어 번역문 자체의 저작권은 피하고, 원문 또는 직접 현대적 재구성이 가능한 작품 중심
- 너무 긴 작품은 일부 장면만 뽑을 수 있게 표시
- 자연, 이별, 기다림, 밤, 비, 길, 강, 계절, 고독, 사랑, 노동, 여행 등 이미지가 있는 작품 우선
- 학습 데이터로 변환하기 좋게 다양성을 확보

출력 형식:
| 번호 | 작품명 | 작가/출처 | 시대 | 핵심 이미지 | 정서 | 데이터화 난이도 | 비고 |

우선 50편만 선정해줘.
선정 후 내가 확인하면 200편까지 확장할 거야.
```

## 실행 프롬프트 3-2: 51~100번

```text
앞서 선정한 기준을 유지해서 고전시 후보 51~100번을 추가 선정해줘.
중복 없이 다양성을 늘려줘.
```

## 실행 프롬프트 3-3: 101~150번

```text
앞서 선정한 기준을 유지해서 고전시 후보 101~150번을 추가 선정해줘.
중복 없이 자연/이별/기다림/밤/길/계절 이미지가 고르게 들어가게 해줘.
```

## 실행 프롬프트 3-4: 151~200번

```text
앞서 선정한 기준을 유지해서 고전시 후보 151~200번을 추가 선정해줘.
너무 유명한 작품에만 치우치지 말고 데이터 다양성을 확보해줘.
```

## 완료 기준

```text
고전시 후보 200편 확보
작품별 이미지, 정서, 난이도 정리
초기 변환 우선순위 후보 표시
```

---

# 4단계. 고전시 기반 experience-poem 데이터 제작

## 목적

고전시 후보를 학습 가능한 `experience → 3줄 poem` 데이터로 변환한다.

## 실행 프롬프트

```text
확률분포 시 생성기 데이터셋 4단계를 진행하자.

목표:
고전시 후보 10편을 학습용 experience-poem pair로 변환한다.

프로젝트 기준:
- 최종 앱은 프롬프트 모드 없이 파인튜닝 모델 단일 모드다.
- 입력은 사용자 경험 프롬프트다.
- 출력은 무조건 3줄 시다.
- 학습 데이터는 experience → poem 구조다.
- poem은 정확히 3줄이어야 한다.
- 각 줄은 짧고 이미지 중심이어야 한다.
- 설명문, 해설문, 보고서체는 금지한다.
- 고전시 원문을 그대로 베끼지 말고 핵심 이미지와 정서를 현대적 3줄 시로 재구성한다.
- experience는 실제 사용자가 입력할 법한 현대 일상 경험 문장이어야 한다.
- 하드코딩 템플릿처럼 같은 구조가 반복되면 안 된다.

이번 변환 대상:
[여기에 10개 작품명/원문/요약 붙여넣기]

각 항목마다 다음을 만들어줘.

필드:
- id
- source_type: classic_poem
- source_title
- source_author
- source_text
- experience
- poem
- style_tags

출력 순서:
1. 변환 표
2. JSONL 형식
3. 품질이 애매한 항목과 이유
4. 다음 검수에서 봐야 할 점

주의:
source_text에는 가능한 한 짧은 원문 또는 원문 확인용 핵심 구절만 넣고,
poem에는 원문 표현을 그대로 복붙하지 말고 직접 재구성한 3줄 시를 넣어줘.
```

## 검수 프롬프트

```text
방금 만든 JSONL 데이터를 검수해줘.

검수 기준:
1. poem이 정확히 3줄인지
2. 각 줄이 너무 길지 않은지
3. experience가 실제 사용자 입력처럼 자연스러운지
4. poem이 설명문/해설문/보고서체가 아닌지
5. 고전시 원문의 핵심 이미지나 정서가 살아 있는지
6. experience와 poem 사이에 연결되는 사물/감각어가 있는지
7. poem의 문장 구조가 서로 너무 반복되지 않는지
8. 원문 표현을 그대로 베낀 느낌이 없는지

출력:
- 통과 항목
- 수정 필요 항목
- 수정된 JSONL
- 전체 데이터셋에 적용할 작성 규칙
```

## 완료 기준

```text
classic_0001부터 순차적으로 JSONL 생성
poem 전부 정확히 3줄
experience 전부 실제 사용자 입력처럼 자연스러움
원문 복붙 느낌 없음
문장 구조 반복 과하지 않음
data/experience_poem_pairs.jsonl에 추가 가능
```

---

# 5단계. 현대 일상 경험 데이터 제작

## 목적

실제 앱 사용자가 입력할 법한 현대 경험을 추가해 고전시 기반 데이터의 거리감을 보완한다.

## 실행 프롬프트

```text
확률분포 시 생성기 학습용 현대 일상 경험 데이터 50개를 만들어줘.

목표:
- 실제 사용자가 입력할 법한 경험 프롬프트
- 각 경험에 대응하는 3줄 한국어 시
- source_type은 modern_daily
- 대학생, 과제, 밤, 비, 지하철, 카페, 창업, 여행, 혼자 있는 시간, 수면, 알바, 앱 개발 같은 현대적 경험 포함
- 너무 감성 카피처럼 만들지 말고, 구체적인 상황에서 출발
- poem은 무조건 3줄
- 각 줄은 짧게
- 설명문 금지
- 경험 속 사물/장소/감각을 최소 1개 이상 보존
- 일상 표현을 살짝 비틀어 언어적 도약을 만들기

출력:
JSONL 형식으로 50개
```

## 추가 데이터 프롬프트

```text
같은 기준으로 modern_daily 데이터 50개를 더 만들어줘.
이번에는 앞선 데이터와 상황이 겹치지 않게 해줘.
혼자 이동, 수업, 병원, 편의점, 새벽 작업, 앱 오류, 휴학, 수영, 운동, 가족, 친구 약속 같은 상황도 포함해줘.
```

## 완료 기준

```text
modern_daily 50~100개 확보
전부 3줄 poem
앱 사용자 입력과 가까운 experience 확보
고전시 데이터와 중복되지 않는 현대 소재 포함
```

---

# 6단계. 데이터셋 검수 스크립트 제작

## 목적

마스터 JSONL을 검수하고 학습용 train.jsonl로 변환하는 스크립트를 만든다.

## 실행 프롬프트

```text
training/prepare_dataset.py 파일을 만들어줘.

목표:
- data/experience_poem_pairs.jsonl을 읽는다.
- 각 row의 experience, poem을 검수한다.
- poem은 반드시 3줄이어야 한다.
- 빈 experience나 빈 poem을 잡는다.
- 너무 긴 행을 잡는다.
- 설명문 어미가 많은 poem을 잡는다.
- 중복 experience를 잡는다.
- 문제가 없는 데이터만 data/train.jsonl로 저장한다.
- 문제 있는 데이터는 data/dataset_issues.jsonl로 저장한다.

학습용 train.jsonl 형식은 다음처럼 만들어줘.

{
  "messages": [
    {
      "role": "system",
      "content": "너는 경험을 3줄 한국어 시로 바꾸는 시 생성 모델이다."
    },
    {
      "role": "user",
      "content": "경험: ..."
    },
    {
      "role": "assistant",
      "content": "1행\n2행\n3행"
    }
  ]
}

Python 코드 전체를 제공해줘.
실행 방법도 같이 써줘.
```

## 완료 기준

```text
prepare_dataset.py 실행 가능
train.jsonl 생성 가능
dataset_issues.jsonl 생성 가능
3줄 검증 포함
중복 검증 포함
```

---

# 7단계. Colab LoRA 학습 노트북 제작

## 목적

Colab T4에서 Qwen/Qwen2.5-0.5B-Instruct를 LoRA 파인튜닝하는 노트북을 만든다.

## 실행 프롬프트

```text
Colab에서 Qwen/Qwen2.5-0.5B-Instruct를 LoRA 파인튜닝하는 노트북 코드를 만들어줘.

조건:
- 학습은 Colab T4 기준
- 데이터셋은 data/train.jsonl 형식
- Hugging Face Hub에서 train.jsonl을 업로드해서 불러오거나, Colab에 직접 업로드해서 불러올 수 있게 둘 다 지원
- base model: Qwen/Qwen2.5-0.5B-Instruct
- LoRA 사용
- target_modules는 우선 q_proj, v_proj
- 필요시 q_proj, k_proj, v_proj, o_proj 확장 가능하게 변수화
- 출력은 adapter 저장
- 학습 후 adapter를 Hugging Face Hub에 업로드
- 가능하면 merged model도 생성해서 업로드
- 3줄 시 생성 테스트 코드 포함

필요 라이브러리:
transformers
datasets
peft
trl
accelerate
bitsandbytes
huggingface_hub

출력:
Colab 셀 단위로 복사 가능한 코드
각 셀의 목적 설명
주의할 점
```

## 완료 기준

```text
train_lora_colab.ipynb 초안 완성
Colab에서 순서대로 실행 가능
adapter 저장 코드 포함
merged model 생성 코드 포함
HF Hub 업로드 코드 포함
```

---

# 8단계. Colab에서 학습 실행 및 오류 해결

## 목적

Colab에서 실제 학습을 실행하고 오류를 해결한다.

## 오류 해결 프롬프트

```text
Colab에서 Qwen2.5-0.5B-Instruct LoRA 학습 중 오류가 났어.

내 환경:
- Colab
- GPU: [T4/P100/없음 중 적기]
- base model: Qwen/Qwen2.5-0.5B-Instruct
- 데이터셋: train.jsonl
- LoRA 사용

오류 로그:
[여기에 전체 오류 로그 붙여넣기]

현재 실행한 코드 셀:
[문제 난 셀 코드 붙여넣기]

원인 분석하고, 수정된 코드 셀을 다시 줘.
가능하면 VRAM 부족 / 라이브러리 버전 문제 / 데이터 포맷 문제 / tokenizer 문제를 구분해서 설명해줘.
```

## 완료 기준

```text
Colab에서 학습 완료
adapter 파일 생성
학습 후 3줄 시 테스트 가능
```

---

# 9단계. Adapter + merged model Hugging Face Hub 업로드

## 목적

학습된 LoRA adapter와 merged model을 Hugging Face Hub에 업로드한다.

## 실행 프롬프트

```text
LoRA 학습이 끝났어.

adapter repo:
[Hugging Face adapter repo ID]

merged model repo:
[Hugging Face merged model repo ID]

테스트 입력:
1. 야자 끝나고 비 오는 정류장에서 버스를 기다림
2. 새벽에 과제를 하다가 모니터 빛에 눈이 아픔
3. 카페에서 창업 아이디어를 정리하다가 갑자기 막힘
4. 혼자 여행지 술집에서 감자튀김과 맥주를 먹음
5. AI 프로젝트를 하다가 언어가 망가지는 걸 봄

이 모델이 3줄 시 형식을 잘 학습했는지 평가하는 테스트 코드를 만들어줘.
Colab에서 실행 가능해야 하고, 출력은 각 입력별 poem, 줄 수, validation_reason을 보여줘.
```

## 완료 기준

```text
adapter Hugging Face Hub 업로드 완료
merged model Hugging Face Hub 업로드 완료
테스트 입력 5개에 대해 3줄 출력 확인
```

---

# 10단계. Hugging Face Space backend 교체

## 목적

Space backend를 파인튜닝 모델 기반 단일 모드로 바꾼다.

## 실행 프롬프트

```text
Hugging Face Space backend용 app.py를 새로 작성해줘.

최종 구조:
- 프롬프트 모드 없음
- mode는 finetuned_experiment 하나만 사용
- 입력: experience, languageJump
- 출력: poem, mode, model, params
- poem은 검증 실패해도 원문 그대로 반환
- validation_status는 valid 또는 invalid_shown
- validation_reason은 ok, not_three_lines, prose_report_like_output, broken_korean_like_output, no_experience_keyword_match 등으로 표시
- 모델은 Hugging Face Hub의 merged model을 우선 로드
- 환경변수 MODEL_ID로 모델 ID를 바꿀 수 있게 함
- Qwen/Qwen2.5-0.5B-Instruct 기반
- custom decoding 유지
- languageJump에 따라 temperature, top_p, topk, remove_top_n, band_size가 바뀜
- 3줄 생성을 목표로 newline_stop = 3
- max_new_tokens는 50~60
- request_id와 elapsed_seconds를 params에 포함
- root / 에서 app_version, model, show_invalid_outputs를 반환

추가:
- Hugging Face 무료 CPU Space에서 돌아가는 것을 우선
- 너무 복잡한 GPU 전용 최적화는 넣지 말 것
- 단, custom decoding은 나중에 KV cache 적용 가능하게 구조를 깔끔하게 작성

출력:
app.py 전체 코드
requirements.txt
Dockerfile
```

## 완료 기준

```text
hf-space-backend/app.py 교체
requirements.txt 교체
Dockerfile 확인
Space root / 에서 최신 app_version 확인
/generate 응답 확인
```

---

# 11단계. Vercel API proxy 수정

## 목적

Vercel API가 `experience`, `languageJump`만 backend로 전달하도록 단순화한다.

## 실행 프롬프트

```text
Vercel의 api/generate-poem.js를 수정해줘.

최종 구조:
- 프론트에서 experience, languageJump만 받는다.
- mode는 더 이상 받지 않거나, 내부에서 finetuned_experiment로 고정한다.
- Hugging Face Space /generate로 요청을 전달한다.
- HF_BACKEND_URL 환경변수를 지원한다.
- Hugging Face backend가 오류를 내도 demo fallback을 만들지 않는다.
- 실패 시에도 가능한 한 backend 응답을 그대로 전달한다.
- poem을 임의로 생성하지 않는다.
- 하드코딩 시 템플릿 금지

출력:
api/generate-poem.js 전체 코드
Vercel 환경변수 설정 방법
테스트용 fetch 코드
```

## 완료 기준

```text
mode 제거
experienceDensity 제거
dadaIntensity 제거
languageJump만 전달
backend 응답 그대로 전달
demo fallback 없음
```

---

# 12단계. Frontend UI 단순화

## 목적

프론트엔드를 단일 모드, 단일 슬라이더 구조로 바꾼다.

## 실행 프롬프트

```text
index.html을 최종 앱 구조에 맞게 수정해줘.

삭제할 것:
- 프롬프트 모드 / 확률 실험 모드 토글
- 경험 밀도 슬라이더
- 다다 강도 슬라이더
- demo fallback 관련 함수
- 실패 시 임의 시 생성 로직

남길 것:
- 경험 입력창
- 언어 도약도 슬라이더
- 생성하기 버튼
- 초기화 버튼
- 3줄 시 출력 영역
- validation/debug 정보 영역

API 요청 payload:
{
  "experience": "...",
  "languageJump": 65
}

출력 영역:
- poem은 그대로 표시
- params.validation_status
- params.validation_reason
- params.strategy
- params.temperature
- params.top_p
- params.topk
- params.remove_top_n
- params.request_id
- params.elapsed_seconds 표시

주의:
- 검증 실패해도 poem을 숨기지 않는다.
- 실패 메시지나 demo poem으로 대체하지 않는다.
- 디자인은 기존 톤을 유지하되 UI는 단순화한다.

출력:
index.html 전체 코드
```

## 완료 기준

```text
모드 토글 제거
슬라이더는 언어 도약도만 남음
출력 영역은 poem + debug params 표시
demo fallback 없음
```

---

# 13단계. 전체 연결 테스트

## 목적

Vercel → API proxy → Hugging Face Space backend 연결을 확인한다.

## Hugging Face 테스트 프롬프트

```text
Hugging Face Space backend 연결을 테스트하려고 해.

내 backend URL:
https://z-unghyun-poem-generator-backend.hf.space

테스트할 endpoint:
/
/generate

요청 payload:
{
  "experience": "야자 끝나고 비 오는 정류장에서 버스를 기다림",
  "languageJump": 65
}

브라우저 콘솔 fetch 코드와 PowerShell curl 코드를 만들어줘.

확인해야 할 것:
- root에서 app_version이 최신인지
- /generate 응답에 poem이 있는지
- params.validation_status가 있는지
- params.validation_reason이 있는지
- mode가 finetuned_experiment인지
- show_invalid_outputs 정책이 반영됐는지

응답 해석 기준도 같이 정리해줘.
```

## Vercel 테스트 프롬프트

```text
Vercel frontend와 Hugging Face Space backend 연결을 테스트하려고 해.

구조:
Vercel index.html
→ /api/generate-poem
→ Hugging Face Space /generate

확인할 것:
- Vercel 환경변수 HF_BACKEND_URL
- api/generate-poem.js가 올바른 backend로 요청하는지
- CORS 문제인지 proxy 문제인지 구분
- 브라우저 Network 탭에서 봐야 할 항목
- console fetch 테스트 코드
- 실패 시 원인별 진단표

내 backend:
https://z-unghyun-poem-generator-backend.hf.space/generate

출력:
1. 브라우저 콘솔 테스트 코드
2. Vercel API 직접 테스트 코드
3. Network 탭 확인법
4. 에러별 원인 진단표
```

## 완료 기준

```text
HF root 최신 버전 확인
HF /generate 응답 확인
Vercel /api/generate-poem 응답 확인
프론트에서 poem과 params 표시 확인
```

---

# 14단계. 생성 효율 개선

## 목적

무료 CPU Space에서도 가능한 수준으로 생성 속도를 개선한다.

## 실행 프롬프트

```text
현재 Hugging Face 무료 CPU Space에서 파인튜닝 모델 + custom decoding으로 3줄 시를 생성하고 있어.

목표:
- 실험 구조는 유지
- 생성 속도 개선
- 출력은 3줄 고정
- custom decoding 유지
- 언어 도약도에 따른 logits 조작 유지

현재 병목 후보:
- 매 토큰마다 전체 generated sequence를 다시 model에 넣음
- max_new_tokens가 큼
- 프롬프트가 김
- adapter 로딩이 느림
- CPU 추론 한계

개선해줘:
1. max_new_tokens 최적값
2. 프롬프트 축소
3. newline_stop = 3 적용
4. KV cache를 custom decoding에 적용하는 방법
5. adapter 방식과 merged model 방식의 속도 비교
6. CPU Space에서 우선 적용할 코드 변경 순서

출력:
- 개선 우선순위
- 수정된 custom decoding 함수
- 속도 측정 로그 코드
```

## 완료 기준

```text
max_new_tokens 50~60
newline_stop = 3
프롬프트 축소
평균 응답 시간 측정
필요 시 KV cache 적용
```

---

# 15단계. 실험 로그/결과 저장

## 목적

생성 결과를 실험 로그로 저장해 데이터셋 개선과 디코딩 개선에 활용한다.

## 실행 프롬프트

```text
생성 결과를 실험 로그로 저장하는 기능을 추가하고 싶어.

목표:
- 사용자가 생성할 때마다 결과를 JSON 형태로 기록
- 최소한 backend 로그에는 남기기
- 가능하면 local file 또는 Hugging Face Space ephemeral storage에 logs.jsonl 저장
- 무료 Space에서 파일 저장이 제한적이면 대안을 제안
- 저장 항목:
  - timestamp
  - request_id
  - experience
  - poem
  - languageJump
  - validation_status
  - validation_reason
  - strategy
  - temperature
  - top_p
  - topk
  - remove_top_n
  - elapsed_seconds

주의:
- 개인정보 민감할 수 있으므로 public 저장은 조심
- 일단 개발 중 디버그용으로만 사용

출력:
1. app.py에 추가할 logging 함수
2. 저장 방식별 장단점
3. 무료 Hugging Face Space에서 현실적인 방식
```

## 완료 기준

```text
request_id별 로그 확인 가능
validation_reason 분포 확인 가능
언어 도약도별 결과 추적 가능
```

---

# 16단계. 데이터셋 개선 루프

## 목적

생성 실패 결과를 바탕으로 데이터셋을 보강한다.

## 실행 프롬프트

```text
아래는 현재 모델의 생성 결과와 validation_reason이야.

목표:
- 어떤 데이터가 부족해서 이런 실패가 나는지 분석
- 데이터셋에 추가해야 할 experience-poem pair 유형 제안
- 기존 decoding 문제인지 데이터 문제인지 구분
- 다음 학습 데이터 보강안을 JSONL로 제시

생성 로그:
[여기에 여러 결과 붙여넣기]

출력:
1. 실패 유형 분류
2. 원인 추정
3. 데이터셋 보강 방향
4. 추가할 JSONL 데이터 20개
5. decoding 파라미터 수정이 필요한 경우 제안
```

## 완료 기준

```text
실패 유형별 원인 정리
보강 데이터 20개 이상 생성
다음 학습 라운드 계획 수립
```

---

# 17단계. 최종 README 작성

## 목적

GitHub repo의 README를 프로젝트 기준에 맞게 정리한다.

## 실행 프롬프트

```text
현재 프로젝트 구조와 구현 내용을 바탕으로 GitHub README.md를 작성해줘.

프로젝트:
확률분포 시 생성기

핵심:
- 경험 입력을 3줄 시로 변환
- Qwen2.5-0.5B-Instruct LoRA 파인튜닝
- 언어 도약도 슬라이더로 decoding 조작
- logits boost / top-k / top-p / anti-greedy sampling
- validation 실패도 숨기지 않고 원문 출력
- Vercel + Hugging Face Space 배포

README 구성:
1. 프로젝트 소개
2. 핵심 아이디어
3. 아키텍처
4. 데이터셋
5. 학습 방법
6. 배포 구조
7. API 사용법
8. validation/debug 정책
9. 실행 방법
10. 한계와 다음 개선점

한국어로 작성해줘.
```

## 완료 기준

```text
README.md 업데이트
프로젝트 목적 명확화
학습/배포/실험 구조 설명 포함
```

---

# 18단계. 운영 체크리스트 정리

## 목적

프로젝트를 계속 진행할 때 확인해야 할 상태를 체크리스트로 유지한다.

## 실행 프롬프트

```text
현재 확률분포 시 생성기 프로젝트의 운영 체크리스트를 정리해줘.

포함할 것:
- 데이터셋 상태
- train.jsonl 상태
- Colab 학습 상태
- adapter repo 상태
- merged model repo 상태
- Hugging Face Space backend 상태
- Vercel frontend/API 상태
- validation/debug 표시 상태
- 남은 TODO
- 다음 작업 우선순위

출력:
1. 체크리스트 표
2. 현재 상태 / 완료 조건 / 다음 액션
3. 다음에 진행할 단계 추천
```

## 완료 기준

```text
현재 상태를 한눈에 확인 가능
다음 작업 단계 명확화
반복 작업 시 기준 문서로 사용 가능
```

---

# 체크포인트

## 체크포인트 1. 데이터셋 준비 완료

완료 조건:

```text
data/experience_poem_pairs.jsonl 존재
data/train.jsonl 존재
총 250~300개
classic_poem 200개
modern_daily 50~100개
모든 poem이 3줄
prepare_dataset.py 통과
dataset_issues.jsonl에서 치명적 오류 없음
```

확인 프롬프트:

```text
data/experience_poem_pairs.jsonl과 data/train.jsonl을 기준으로 데이터셋 준비 상태를 점검해줘.
총 개수, source_type 분포, 3줄 검증, 중복 experience, 문제 항목을 확인하고 체크포인트 1 통과 여부를 판단해줘.
```

---

## 체크포인트 2. Colab 학습 완료

완료 조건:

```text
LoRA adapter 생성
merged model 생성
Hugging Face Hub 업로드 완료
테스트 입력 5개에서 3줄 출력 확인
validation_reason 확인 가능
```

확인 프롬프트:

```text
Colab 학습 결과를 점검해줘.
adapter repo, merged model repo, 테스트 출력 5개를 기준으로 체크포인트 2 통과 여부를 판단하고, 문제가 있으면 원인을 분류해줘.
```

---

## 체크포인트 3. Backend 교체 완료

완료 조건:

```text
Hugging Face Space root / 응답에 최신 app_version 표시
/generate 응답에 poem, params 표시
mode = finetuned_experiment
validation 실패해도 poem 원문 출력
show_invalid_outputs = true
```

확인 프롬프트:

```text
Hugging Face Space backend 상태를 점검해줘.
root / 응답과 /generate 응답을 기준으로 최신 app_version, model ID, mode, poem, params, show_invalid_outputs 정책이 맞는지 확인해줘.
```

---

## 체크포인트 4. Frontend 교체 완료

완료 조건:

```text
프롬프트 모드 토글 없음
경험 밀도 슬라이더 없음
다다 강도 슬라이더 없음
언어 도약도 슬라이더 하나만 있음
3줄 시 출력
debug params 표시
demo fallback 없음
```

확인 프롬프트:

```text
Vercel frontend 상태를 점검해줘.
index.html과 브라우저 화면 기준으로 프롬프트 모드 토글 제거, 언어 도약도 단일 슬라이더, poem 원문 출력, debug params 표시, demo fallback 제거 여부를 확인해줘.
```

---

## 체크포인트 5. 최적화 완료

완료 조건:

```text
max_new_tokens 50~60
newline_stop = 3
평균 응답 시간 측정
CPU Space에서 사용 가능한 수준 확인
필요 시 KV cache 적용
```

확인 프롬프트:

```text
생성 효율 상태를 점검해줘.
현재 max_new_tokens, newline_stop, 평균 응답 시간, custom decoding 방식, KV cache 적용 여부를 기준으로 체크포인트 5 통과 여부를 판단해줘.
```

---

# 단계별 진행 규칙

이 문서를 사용할 때는 아래처럼 지시한다.

```text
execution-prompts.md 기준으로 4단계만 진행하자.
이번 목표는 고전시 10편을 experience-poem JSONL로 변환하는 거야.
```

또는:

```text
execution-prompts.md 기준으로 체크포인트 1을 점검하자.
현재 data/experience_poem_pairs.jsonl 내용을 기준으로 통과 여부를 판단해줘.
```

---

# 현재 권장 다음 단계

현재 1~3단계가 완료되었고, 다음은 4단계다.

권장 시작 묶음:

```text
1. 공무도하가
2. 황조가
3. 정읍사
4. 가시리
5. 청산별곡
6. 청산리 벽계수야
7. 동짓달 기나긴 밤을
8. 이화우 흩뿌릴 제
9. 송인
10. 추야우중
```

이 10개로 첫 변환 톤을 잡은 뒤, 검수 기준을 확정하고 20개 단위로 확장한다.
