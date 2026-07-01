import os
import re
from typing import Dict, List, Literal

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "120"))
EXPERIMENT_MAX_NEW_TOKENS = int(os.getenv("EXPERIMENT_MAX_NEW_TOKENS", "80"))
EXPERIMENT_TOPK = int(os.getenv("EXPERIMENT_TOPK", "220"))

app = FastAPI(title="Poem Generator Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    mode: Literal["prompt", "experiment"] = "prompt"
    experience: str = ""
    experienceDensity: int = Field(default=80, ge=0, le=100)
    languageJump: int = Field(default=65, ge=0, le=100)
    dadaIntensity: int = Field(default=40, ge=0, le=100)


class GenerateResponse(BaseModel):
    poem: str
    mode: str
    model: str
    params: Dict[str, float | int | str]


print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
if not torch.cuda.is_available():
    model.to("cpu")
model.eval()
DEVICE = next(model.parameters()).device


FORBIDDEN_MARKERS = [
    "[삶의 경험]",
    "[슬라이더]",
    "[생성 규칙]",
    "[경험어 후보]",
    "[실험 조건]",
    "시 본문:",
    "해설",
    "제목:",
]


def slider_tone(value: int, low: str, mid: str, high: str) -> str:
    if value < 35:
        return low
    if value < 70:
        return mid
    return high


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```.*?\n", "", text, flags=re.DOTALL)
    text = text.replace("```", "").strip()

    # If the model starts continuing the prompt template, cut that leakage away.
    for marker in FORBIDDEN_MARKERS:
        if marker in text:
            text = text.split(marker)[0].strip()

    text = re.sub(r"^(제목|시|본문)\s*[:：].*\n", "", text).strip()
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if any(marker in line for marker in FORBIDDEN_MARKERS):
            continue
        if line.startswith(("- 경험", "- 언어", "- 다다", "1.", "2.", "3.", "4.", "5.")):
            continue
        lines.append(line)
    return "\n".join(lines[:10])


def extract_keywords(text: str, limit: int = 12) -> List[str]:
    fallback = ["비", "정류장", "버스", "기다림", "가방", "불빛"]
    words = re.sub(r"[.,!?;:()\[\]{}\"'“”‘’]", " ", text)
    words = [word.strip() for word in words.split() if len(word.strip()) > 1]
    seen = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:limit] or fallback


def is_invalid_poem(poem: str, experience: str) -> bool:
    if not poem.strip():
        return True
    if any(marker in poem for marker in FORBIDDEN_MARKERS):
        return True
    if "나는 한반도" in poem or "애드바이토르" in poem or "일일 수업" in poem:
        return True

    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    if len(lines) < 3:
        return True

    # A single long prose paragraph is not useful for this interface.
    if len(lines) <= 2 and len(poem) > 140:
        return True

    experience_keywords = set(extract_keywords(experience, limit=6))
    poem_text = poem.replace(" ", "")
    matched = sum(1 for word in experience_keywords if word.replace(" ", "") in poem_text)
    return matched == 0 and bool(experience.strip())


def repair_poem_from_experience(req: GenerateRequest) -> str:
    words = extract_keywords(req.experience, limit=6)
    first = words[0] if len(words) > 0 else "비"
    second = words[1] if len(words) > 1 else "정류장"
    third = words[2] if len(words) > 2 else "버스"
    fourth = words[3] if len(words) > 3 else "기다림"

    if req.languageJump >= 70:
        images = ["젖은 불빛", "유리의 숨", "늦은 바퀴", "접힌 이름"]
    elif req.languageJump >= 40:
        images = ["번지는 가로등", "차가운 손바닥", "흔들리는 창문", "젖은 신발"]
    else:
        images = ["비 오는 밤", "조용한 정류장", "늦은 버스", "작은 기다림"]

    lines = [
        f"{first}이 {second} 위에 조용히 내려앉는다",
        f"나는 {third}를 기다리며 {images[0]}을 바라본다",
        f"젖은 시간은 주머니 속에서 조금씩 구겨지고",
        f"{fourth}은 오지 않는 바퀴 소리처럼 길어진다",
        f"가로등 아래 내 이름이 잠깐 흐려진다",
        f"밤은 아직 도착하지 않은 문장으로 서 있다",
    ]

    if req.dadaIntensity >= 70:
        lines.insert(3, f"{first} / {second} / {third}")
    if req.dadaIntensity >= 85:
        lines.append(f"도착 / 미도착 / 다시 {fourth}")

    return "\n".join(lines[:9])


def apply_chat_template(system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"[system]\n{system}\n\n[user]\n{user}\n\n[assistant]\n"


def build_prompt_mode_prompt(req: GenerateRequest) -> str:
    experience_rule = slider_tone(
        req.experienceDensity,
        "입력 경험을 배경으로만 살짝 반영한다.",
        "입력 경험의 장소, 사물, 감정을 여러 행에 반영한다.",
        "입력 경험의 핵심 장면과 감각을 시 전체의 중심으로 삼는다.",
    )
    jump_rule = slider_tone(
        req.languageJump,
        "문장은 비교적 자연스럽고 서정적으로 쓴다.",
        "평범한 표현을 피하고 낯선 은유를 섞는다.",
        "관습적 연결을 피하고 강한 이미지의 비약을 만든다.",
    )
    dada_rule = slider_tone(
        req.dadaIntensity,
        "문법과 의미는 대부분 보존한다.",
        "일부 반복과 행갈이를 사용하되 전체 의미는 유지한다.",
        "반복, 절단, 비문을 허용하되 입력 경험과 연결을 유지한다.",
    )
    experience = req.experience.strip() or "비어 있는 경험"
    system = (
        "너는 한국어 현대시 생성기다. "
        "반드시 시 본문만 출력한다. 제목, 해설, 목록, 대괄호 섹션, 자기소개, 다른 경험 설명을 절대 쓰지 않는다. "
        "입력된 경험에 없는 새로운 배경이나 신상 정보를 만들지 않는다."
    )
    user = f"""입력 경험: {experience}

이 경험만 바탕으로 6행의 한국어 시를 써라.
각 행은 짧게 쓰고 반드시 줄바꿈한다.
첫 행부터 바로 시를 시작한다.
금지: 제목, 해설, [삶의 경험], [생성 규칙], 산문 설명, 자기소개.

조건:
- {experience_rule}
- {jump_rule}
- {dada_rule}
- 장소/사물/감각어를 사용한다.
- 입력 경험의 핵심어를 최소 2개 포함한다.

시:"""
    return apply_chat_template(system, user)


def build_experiment_seed_prompt(req: GenerateRequest) -> str:
    keywords = ", ".join(extract_keywords(req.experience))
    experience = req.experience.strip() or "비어 있는 경험"
    system = "너는 한국어 실험시 생성기다. 출력은 시 본문만 작성한다. 제목과 해설은 쓰지 않는다."
    user = f"""입력 경험: {experience}
경험어: {keywords}

위 경험어만 붙잡고 6~8행의 한국어 실험시를 써라.
첫 행부터 바로 시를 시작한다.

시:"""
    return apply_chat_template(system, user)


def generate_prompt_mode(req: GenerateRequest) -> str:
    prompt = build_prompt_mode_prompt(req)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    temperature = min(0.82, 0.48 + req.languageJump * 0.004)
    top_p = 0.82
    repetition_penalty = 1.12 + max(0, 70 - req.dadaIntensity) * 0.003

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    poem = clean_text(tokenizer.decode(generated, skip_special_tokens=True))
    if is_invalid_poem(poem, req.experience):
        return repair_poem_from_experience(req)
    return poem


def keyword_token_ids(keywords: List[str]) -> List[int]:
    ids = set()
    for keyword in keywords:
        for variant in [keyword, " " + keyword, "\n" + keyword]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            ids.update(encoded)
    return list(ids)


def apply_dada_postprocess(poem: str, dada_intensity: int) -> str:
    if dada_intensity < 60:
        return poem
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    if len(lines) < 3:
        return poem
    if dada_intensity >= 85:
        fractured = []
        for index, line in enumerate(lines):
            fractured.append(line)
            words = line.split()
            if index % 2 == 0 and len(words) >= 2:
                fractured.append(f"{words[0]} / {words[-1]}")
        return "\n".join(fractured[:12])
    lines.insert(2, lines[1])
    return "\n".join(lines[:11])


def generate_experiment_mode(req: GenerateRequest) -> str:
    prompt = build_experiment_seed_prompt(req)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    prompt_length = input_ids.shape[-1]

    keywords = extract_keywords(req.experience)
    boost_ids = keyword_token_ids(keywords)
    boost_value = 0.6 + req.experienceDensity * 0.035

    remove_top_n = int(2 + req.languageJump * 0.28)
    band_size = int(25 + req.languageJump * 1.35)
    temperature = 0.7 + req.languageJump * 0.012
    topk = max(remove_top_n + 5, min(EXPERIMENT_TOPK, model.config.vocab_size))

    generated = input_ids

    with torch.no_grad():
        for _ in range(EXPERIMENT_MAX_NEW_TOKENS):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :].float()

            if boost_ids:
                valid_ids = [token_id for token_id in boost_ids if token_id < logits.shape[-1]]
                logits[:, valid_ids] += boost_value

            if tokenizer.eos_token_id is not None:
                logits[:, tokenizer.eos_token_id] -= 0.2

            logits = logits / max(temperature, 0.1)

            topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
            start = min(remove_top_n, topk_indices.shape[-1] - 2)
            end = min(start + band_size, topk_indices.shape[-1])
            candidate_indices = topk_indices[:, start:end]
            candidate_logits = torch.gather(logits, 1, candidate_indices)
            probs = torch.softmax(candidate_logits, dim=-1)
            sampled_pos = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(candidate_indices, 1, sampled_pos)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            decoded_so_far = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
            if decoded_so_far.count("\n") >= 7:
                break

    decoded = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
    poem = apply_dada_postprocess(clean_text(decoded), req.dadaIntensity)
    if is_invalid_poem(poem, req.experience):
        return repair_poem_from_experience(req)
    return poem


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Poem Generator Backend is running.",
        "model": MODEL_ID,
        "endpoint": "/generate",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.mode == "experiment":
        poem = generate_experiment_mode(req)
        params = {
            "mode": "custom_decoding",
            "experience_boost": round(0.6 + req.experienceDensity * 0.035, 3),
            "remove_top_n": int(2 + req.languageJump * 0.28),
            "band_size": int(25 + req.languageJump * 1.35),
            "topk": EXPERIMENT_TOPK,
            "max_new_tokens": EXPERIMENT_MAX_NEW_TOKENS,
            "newline_stop": 7,
            "temperature": round(0.7 + req.languageJump * 0.012, 3),
        }
    else:
        poem = generate_prompt_mode(req)
        params = {
            "mode": "prompt_instruction",
            "temperature": round(min(0.82, 0.48 + req.languageJump * 0.004), 3),
            "top_p": 0.82,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

    return GenerateResponse(
        poem=poem,
        mode=req.mode,
        model=MODEL_ID,
        params=params,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
