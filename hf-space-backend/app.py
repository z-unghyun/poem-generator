import os
import re
from typing import Dict, List, Literal

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "180"))
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
    text = re.sub(r"^(제목|시|본문)\s*[:：].*\n", "", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:14])


def extract_keywords(text: str, limit: int = 12) -> List[str]:
    fallback = ["비어", "침묵", "빈칸", "기억", "불빛", "정류장"]
    words = re.sub(r"[.,!?;:()\[\]{}\"'“”‘’]", " ", text)
    words = [word.strip() for word in words.split() if len(word.strip()) > 1]
    seen = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:limit] or fallback


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
        "사용자의 경험은 은은한 배경으로만 반영한다.",
        "사용자의 경험 속 장소, 감정, 사물을 여러 행에 반영한다.",
        "사용자의 경험 속 핵심 이미지, 감각, 정서를 시 전체의 중심축으로 삼는다.",
    )
    jump_rule = slider_tone(
        req.languageJump,
        "문장은 비교적 자연스럽고 서정적으로 유지한다.",
        "평범한 표현을 피하고 낯선 은유와 감각적 비약을 사용한다.",
        "관습적 연결을 적극적으로 피하고, 의미가 닿을 듯 말 듯한 강한 언어적 도약을 만든다.",
    )
    dada_rule = slider_tone(
        req.dadaIntensity,
        "문법과 의미는 대부분 보존한다.",
        "일부 반복, 절단, 행갈이를 활용하되 전체 의미는 유지한다.",
        "반복, 절단, 비문, 우연적 조합을 적극 허용한다. 단, 완전한 무의미로 붕괴시키지는 않는다.",
    )
    experience = req.experience.strip() or "구체적인 경험이 입력되지 않았다. 비어 있음 자체를 경험으로 삼아 시를 쓴다."
    system = "너는 한국어 현대시와 실험시를 생성하는 AI 시 생성기다. 출력은 오직 시 본문만 작성한다. 해설, 제목, 따옴표, 목록은 쓰지 않는다."
    user = f"""다음 삶의 경험을 바탕으로 6~10행의 한국어 시를 써라.

[삶의 경험]
{experience}

[슬라이더]
- 경험 밀도: {req.experienceDensity}/100
- 언어 도약도: {req.languageJump}/100
- 다다 강도: {req.dadaIntensity}/100

[생성 규칙]
1. {experience_rule}
2. {jump_rule}
3. {dada_rule}
4. 추상적인 설명보다 이미지, 감각, 사물, 움직임을 중심으로 쓴다.
5. 출력은 시 본문만 작성한다."""
    return apply_chat_template(system, user)


def build_experiment_seed_prompt(req: GenerateRequest) -> str:
    keywords = ", ".join(extract_keywords(req.experience))
    experience = req.experience.strip() or "구체적인 경험이 입력되지 않았다. 비어 있음 자체를 경험으로 삼아 시를 쓴다."
    system = "너는 한국어 실험시 생성기다. 평범한 다음 표현을 피하고 경험어의 의미장을 따라 낯선 행들을 만든다. 출력은 시 본문만 작성한다."
    user = f"""다음 경험에서 출발해 6~10행의 한국어 실험시를 시작하라.

[삶의 경험]
{experience}

[경험어 후보]
{keywords}

[실험 조건]
경험 밀도 {req.experienceDensity}/100, 언어 도약도 {req.languageJump}/100, 다다 강도 {req.dadaIntensity}/100

시 본문:"""
    return apply_chat_template(system, user)


def generate_prompt_mode(req: GenerateRequest) -> str:
    prompt = build_prompt_mode_prompt(req)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    temperature = 0.55 + req.languageJump * 0.006
    top_p = 0.88
    repetition_penalty = 1.05 + max(0, 70 - req.dadaIntensity) * 0.003

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return clean_text(tokenizer.decode(generated, skip_special_tokens=True))


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

            # Special tokens should not dominate poetic generation.
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
    return apply_dada_postprocess(clean_text(decoded), req.dadaIntensity)


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
            "temperature": round(0.55 + req.languageJump * 0.006, 3),
            "top_p": 0.88,
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
