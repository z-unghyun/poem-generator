# 확률분포 시 생성기 프로젝트 계획

## 0. 최종 방향

이 프로젝트는 프롬프트 엔지니어링 기반 시 생성기가 아니라, **경험 기반 3줄 시 형식을 학습한 파인튜닝 모델의 다음 토큰 확률분포를 조작해 시적 언어의 도약을 실험하는 생성기**다.

핵심 질문은 다음과 같다.

```text
LLM이 학습한 평균적 언어 확률분포에서 벗어나도록 만들면,
그 결과는 시적 도약이 되는가,
아니면 산문 회귀나 언어 붕괴가 되는가?
```

## 1. 고정 프로젝트 기준

- 프롬프트 모드는 삭제한다.
- 최종 앱은 파인튜닝된 모델 기반 단일 모드만 가진다.
- 입력은 사용자 경험 프롬프트다.
- 출력은 무조건 3줄 시다.
- 슬라이더는 `언어 도약도` 하나만 남긴다.
- 데이터셋은 저작권 없는 고전시 200개 + 현대 일상 경험 50~100개로 만든다.
- 기본 모델은 `Qwen/Qwen2.5-0.5B-Instruct`다.
- Colab에서 LoRA 파인튜닝한다.
- adapter와 merged model을 둘 다 Hugging Face Hub에 업로드한다.
- Hugging Face Space는 무료 CPU Space에서도 돌아가는 것을 우선한다.
- Vercel frontend → Vercel API proxy → Hugging Face Space backend 구조를 사용한다.
- validation은 출력을 숨기지 않는다.
- 검증 실패해도 원문을 그대로 보여주고 `validation_status`, `validation_reason`만 표시한다.
- 하드코딩 시 템플릿, demo fallback, 실패 출력 숨김은 금지한다.

## 2. 최종 앱 구조

```text
사용자 경험 입력
↓
파인튜닝된 3줄 시 모델
↓
모델의 다음 토큰 logits 계산
↓
언어 도약도에 따라 decoding 조작
↓
3줄 시 출력
↓
validation/debug 라벨 표시
```

## 3. UI 구성

최종 UI는 단일 모드만 제공한다.

```text
[경험 입력창]
[언어 도약도 슬라이더]
[생성하기]
[3줄 시 출력]
[진단 정보]
- validation_status
- validation_reason
- strategy
- temperature
- top_p
- topk
- remove_top_n
- request_id
```

삭제 대상 UI:

```text
프롬프트 모드 / 확률 실험 모드 토글
경험 밀도 슬라이더
다다 강도 슬라이더
의미 보존도 슬라이더
감각화 슬라이더
```

## 4. 데이터셋 설계

초기 데이터셋 규모:

```text
고전시 기반 experience-poem pair: 200개
현대 일상 경험 기반 pair: 50~100개
총 250~300개
```

권장 JSONL 포맷:

```json
{"experience":"비 오는 밤 버스 정류장에서 혼자 기다림","poem":"정류장은 젖은 불빛을 들고\n버스는 아직 오지 않은 문장처럼 멀다\n나는 비의 안쪽에서 조금 늦어진다"}
```

확장 포맷:

```json
{
  "experience": "밤에 혼자 강가를 걸으며 물 위의 달빛을 봄",
  "poem": "강은 달을 접어 주머니에 넣고\n나는 젖은 빛을 밟으며 걷는다\n밤은 발목에서 조용히 풀린다",
  "source_type": "classic_poem",
  "source_title": "원본 고전시 제목",
  "style_tags": ["자연", "고독", "감각", "은유"]
}
```

## 5. 학습 목표

우선순위:

```text
1순위: 3줄 시 형식 안정화
2순위: 입력 경험어와 감각 보존
3순위: 일상 언어를 비트는 시적 도약 생성
```

## 6. 모델 및 학습

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training: Google Colab T4
- Method: LoRA 또는 QLoRA
- Initial target modules: `q_proj`, `v_proj`
- 필요 시 확장: `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

산출물:

```text
1. LoRA adapter
2. merged model
```

## 7. 배포 구조

```text
Vercel frontend
↓
Vercel API proxy: api/generate-poem.js
↓
Hugging Face Space backend: hf-space-backend/app.py
↓
finetuned merged model 또는 base model + LoRA adapter
↓
custom decoding
↓
3줄 시 응답
```

무료 CPU Space 우선 전략:

- 0.5B 모델 유지
- `max_new_tokens` 45~60 수준으로 제한
- 출력 3줄 고정
- 프롬프트 길이 최소화
- merged model 로딩 우선
- custom decoding에 KV cache 적용 검토

## 8. Validation 정책

검증은 출력 차단이 아니라 진단이다.

```text
검증 성공 → 원문 출력 + valid 라벨
검증 실패 → 원문 출력 + invalid_shown 라벨
```

절대 하지 않는 것:

```text
검증 실패 시 실패 메시지로 대체하지 않는다.
검증 실패 시 하드코딩 템플릿으로 보정하지 않는다.
검증 실패 시 결과를 숨기지 않는다.
```

권장 응답 구조:

```json
{
  "poem": "생성된 원문 1행\n생성된 원문 2행\n생성된 원문 3행",
  "mode": "finetuned_experiment",
  "model": "사용한 모델 ID",
  "params": {
    "app_version": "...",
    "request_id": "...",
    "validation_status": "valid 또는 invalid_shown",
    "validation_reason": "ok 또는 실패 사유",
    "strategy": "guided_top_p 또는 anti_greedy",
    "language_jump": 65,
    "temperature": 0.62,
    "top_p": 0.84,
    "topk": 130,
    "remove_top_n": 4,
    "elapsed_seconds": 2.31
  }
}
```

## 9. 단계별 TODO

### Phase 1. 프로젝트 구조 정리

- 프롬프트 모드 삭제 결정 반영
- UI에서 모드 토글 제거
- 슬라이더를 `언어 도약도` 하나로 축소
- 출력 영역을 3줄 시 중심으로 정리
- validation/debug 영역 유지

### Phase 2. 데이터셋 제작

- 저작권 없는 고전시 200편 선정
- 각 시의 핵심 이미지/정서 추출
- 현대적 3줄 시로 재구성
- 경험 프롬프트 역생성
- 현대 일상 경험 50~100개 추가 작성
- JSONL 데이터셋 정리

### Phase 3. LoRA 학습

- Colab 학습 노트북 작성
- Qwen2.5-0.5B-Instruct 로드
- dataset formatting
- LoRA 설정
- 학습 실행
- adapter 저장
- merged model 생성
- Hugging Face Hub 업로드

### Phase 4. 백엔드 교체

- `app.py`에서 파인튜닝 모델 로드
- 프롬프트 모드 함수 제거
- custom decoding 함수 정리
- 3줄 stop 조건 적용
- validation은 원문 출력 + 라벨 정책으로 유지
- request_id/logging 유지

### Phase 5. 생성 효율 개선

- `max_new_tokens` 50~60으로 축소
- 프롬프트 길이 축소
- CPU Space 응답 시간 측정
- 필요 시 KV cache 적용
- adapter vs merged model 로딩 속도 비교

### Phase 6. 실험 기록

- 언어 도약도별 결과 저장
- validation_reason 분포 확인
- 산문 회귀/토큰 붕괴/시적 도약 사례 분류
- 데이터셋 개선 방향 도출

## 10. 최종 방향 한 문장

이 프로젝트는 **저작권 없는 시와 현대 일상 경험으로 학습한 3줄 시 생성 모델의 다음 토큰 확률분포를 조작해, 일상 언어에서 시적 언어로 도약하는 과정과 실패 양상을 함께 관찰하는 실험형 웹 생성기**다.
