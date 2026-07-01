# 시 생성기: AI시대의 시와 스토리텔링

동일한 Hugging Face 오픈소스 언어모델을 두 가지 생성 방식으로 작동시키는 시 생성기입니다.

- **프롬프트 모드**: 슬라이더 값을 자연어 지시문으로 변환해 같은 모델에 전달합니다.
- **확률 실험 모드**: 같은 모델의 logits와 decoding 과정을 직접 조작해 평균적인 단어 선택을 교란합니다.

## 구조

```txt
Browser
↓
Vercel: index.html + /api/generate-poem.js
↓
Hugging Face Space: /generate
↓
Qwen 계열 오픈소스 모델
```

## Vercel 환경변수

Public Hugging Face Space를 쓴다면 환경변수 없이도 기본 주소로 연결됩니다.

기본 backend URL:

```txt
https://z-unghyun-poem-generator-backend.hf.space/generate
```

원하면 Vercel Project Settings → Environment Variables에 아래 값을 넣어 backend를 바꿀 수 있습니다.

```txt
HF_BACKEND_URL=https://z-unghyun-poem-generator-backend.hf.space/generate
```

Private Space를 쓸 경우:

```txt
HF_BACKEND_TOKEN=your_huggingface_token
```

## Hugging Face Space 설정

`hf-space-backend` 폴더의 두 파일을 Hugging Face Space 루트에 업로드하세요.

```txt
app.py
requirements.txt
```

추천 Space 설정:

```txt
SDK: Gradio 또는 Docker
Hardware: Free CPU로 시작
Visibility: Public
```

무료 CPU에서는 모델 로딩과 생성이 느릴 수 있습니다. 우선 기본 모델은 가벼운 모델로 설정되어 있습니다.

```txt
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

Space Settings → Variables에서 더 큰 모델로 변경할 수 있습니다.

```txt
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
```

단, 1.5B 모델은 무료 CPU에서 느릴 수 있습니다.

## API 요청 형식

```json
{
  "mode": "prompt",
  "experience": "비 오는 정류장에서 혼자 버스를 기다렸던 기억",
  "experienceDensity": 80,
  "languageJump": 65,
  "dadaIntensity": 40
}
```

`mode`는 아래 둘 중 하나입니다.

```txt
prompt
experiment
```

## 슬라이더 의미

| 슬라이더 | 프롬프트 모드 | 확률 실험 모드 |
|---|---|---|
| 경험 밀도 | 경험 반영을 자연어로 지시 | 경험어 token logit boost |
| 언어 도약도 | 낯선 은유를 쓰도록 지시 | 상위 확률 token 회피, rank-band sampling |
| 다다 강도 | 반복/절단/비문을 쓰도록 지시 | 반복·절단 후처리 강화 |

## 발표용 핵심 설명

이 생성기는 같은 오픈소스 언어모델을 사용하되, 하나는 프롬프트로 시적 조건을 지시하고 다른 하나는 확률분포와 디코딩 과정을 직접 조작한다. 따라서 두 결과의 차이는 모델 차이가 아니라 생성 방식의 차이로 해석할 수 있다.
