import os
import re
import time
import uuid
from typing import Dict, List, Literal, Tuple

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_VERSION = "stable-experiment-decoding-2026-07-01-1"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "120"))
EXPERIMENT_MAX_NEW_TOKENS = int(os.getenv("EXPERIMENT_MAX_NEW_TOKENS", "80"))
EXPERIMENT_TOPK = int(os.getenv("EXPERIMENT_TOPK", "180"))
FAILURE_MESSAGE = "생성 실패: 모델 출력이 입력 경험과 무관하거나 시 형식을 유지하지 못했습니다. 슬라이더를 낮추고 다시 시도해주세요."

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


print(f"[startup] app_version={APP_VERSION} model={MODEL_ID}", flush=True)
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
VOCAB_SIZE = getattr(model.config, "vocab_size", len(tokenizer))

MARKERS = ["[삶의 경험]", "[슬라이더]", "[생성 규칙]", "[경험어 후보]", "[실험 조건]", "시 본문:", "해설", "제목:"]
BAD_SNIPPETS = ["조용히 내려앉는다", "번지는 가로등", "주머니 속에서 조금씩 구겨지고", "바퀴 소리처럼 길어진다", "도착하지 않은 문장"]


def log_event(request_id: str, message: str) -> None:
    print(f"[poem-generator][{request_id}] {message}", flush=True)


def slider_tone(value: int, low: str, mid: str, high: str) -> str:
    if value < 35:
        return low
    if value < 70:
        return mid
    return high


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```.*?\n", "", text, flags=re.DOTALL).replace("```", "").strip()
    for marker in MARKERS:
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
        if any(marker in line for marker in MARKERS):
            continue
        if line.startswith(("- 경험", "- 언어", "- 다다", "1.", "2.", "3.", "4.", "5.")):
            continue
        lines.append(line)
    return "\n".join(lines[:10])


def normalize_keyword(word: str) -> List[str]:
    word = word.strip()
    if not word:
        return []
    candidates = [word]
    suffixes = ["에서", "에게", "으로", "처럼", "까지", "부터", "하고", "이며", "이랑", "랑", "은", "는", "이", "가", "을", "를", "의", "에"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            candidates.append(word[: -len(suffix)])
    for anchor in ["비", "버스", "정류장", "야자", "기다림", "가방", "밤", "가로등", "불빛", "학교", "집"]:
        if anchor in word:
            candidates.append(anchor)
    seen = []
    for item in candidates:
        if item and item not in seen:
            seen.append(item)
    return seen


def extract_keywords(text: str, limit: int = 12) -> List[str]:
    fallback = ["비", "정류장", "버스", "기다림", "가방", "불빛"]
    words = re.sub(r"[.,!?;:()\[\]{}\"'“”‘’]", " ", text)
    raw_words = [word.strip() for word in words.split() if len(word.strip()) > 1]
    seen = []
    for word in raw_words:
        for candidate in normalize_keyword(word):
            if candidate not in seen:
                seen.append(candidate)
    return seen[:limit] or fallback


def looks_like_broken_korean(poem: str) -> bool:
    compact = poem.replace(" ", "").replace("\n", "")
    if len(compact) < 20:
        return False
    # Very lightweight guard: outputs with almost no common Korean particles/endings
    # often indicate token-level collapse rather than poetic language.
    common = ["은", "는", "이", "가", "을", "를", "에", "의", "고", "다", "서", "로", "와", "과", "도"]
    common_count = sum(compact.count(item) for item in common)
    return common_count <= 2 and len(compact) > 45


def validate_poem(poem: str, experience: str) -> Tuple[bool, str]:
    if not poem.strip():
        return False, "empty_output"
    if any(marker in poem for marker in MARKERS):
        return False, "prompt_marker_leak"
    if any(snippet in poem for snippet in BAD_SNIPPETS):
        return False, "removed_template_pattern"
    if "나는 한반도" in poem or "애드바이토르" in poem or "일일 수업" in poem:
        return False, "unrelated_hallucination"
    if looks_like_broken_korean(poem):
        return False, "broken_korean_like_output"
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    if len(lines) < 3:
        return False, "too_few_lines"
    if len(lines) <= 2 and len(poem) > 140:
        return False, "prose_paragraph"
    if experience.strip():
        keywords = set(extract_keywords(experience, limit=8))
        poem_text = poem.replace(" ", "")
        matched = [word for word in keywords if word.replace(" ", "") in poem_text]
        if not matched:
            return False, "no_experience_keyword_match"
    return True, "ok"


def apply_chat_template(system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"[system]\n{system}\n\n[user]\n{user}\n\n[assistant]\n"


def build_prompt_mode_prompt(req: GenerateRequest) -> str:
    experience_rule = slider_tone(req.experienceDensity, "입력 경험을 배경으로만 살짝 반영한다.", "입력 경험의 장소, 사물, 감정을 여러 행에 반영한다.", "입력 경험의 핵심 장면과 감각을 시 전체의 중심으로 삼는다.")
    jump_rule = slider_tone(req.languageJump, "문장은 비교적 자연스럽고 서정적으로 쓴다.", "평범한 표현을 피하고 낯선 은유를 섞는다.", "관습적 연결을 피하고 강한 이미지의 비약을 만든다.")
    dada_rule = slider_tone(req.dadaIntensity, "문법과 의미는 대부분 보존한다.", "일부 반복과 행갈이를 사용하되 전체 의미는 유지한다.", "반복, 절단, 비문을 허용하되 입력 경험과 연결을 유지한다.")
    experience = req.experience.strip() or "비어 있는 경험"
    system = "너는 한국어 현대시 생성기다. 반드시 시 본문만 출력한다. 제목, 해설, 목록, 대괄호 섹션, 자기소개, 다른 경험 설명을 쓰지 않는다. 입력 경험에 없는 새로운 배경을 만들지 않는다."
    user = f"""입력 경험: {experience}

이 경험만 바탕으로 6행의 한국어 시를 써라.
각 행은 짧게 쓰고 반드시 줄바꿈한다.
첫 행부터 바로 시를 시작한다.
금지: 제목, 해설, 산문 설명, 자기소개.

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
    system = "너는 한국어 실험시 생성기다. 출력은 시 본문만 작성한다. 제목과 해설은 쓰지 않는다. 깨진 글자나 무관한 단어 나열이 아니라 한국어 문장으로 쓴다."
    user = f"""입력 경험: {experience}
경험어: {keywords}

위 경험어를 중심으로 5~7행의 한국어 실험시를 써라.
첫 행부터 바로 시를 시작한다.

시:"""
    return apply_chat_template(system, user)


def generate_prompt_once(req: GenerateRequest, temperature: float, top_p: float) -> str:
    prompt = build_prompt_mode_prompt(req)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    repetition_penalty = 1.12 + max(0, 70 - req.dadaIntensity) * 0.003
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=3, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return clean_text(tokenizer.decode(generated, skip_special_tokens=True))


def generate_prompt_mode(req: GenerateRequest, request_id: str) -> Tuple[str, str, str]:
    first_temperature = min(0.82, 0.48 + req.languageJump * 0.004)
    first_poem = generate_prompt_once(req, temperature=first_temperature, top_p=0.82)
    valid, reason = validate_poem(first_poem, req.experience)
    log_event(request_id, f"prompt first valid={valid} reason={reason} preview={first_poem[:140]!r}")
    if valid:
        return first_poem, "first_pass", reason
    retry_poem = generate_prompt_once(req, temperature=0.42, top_p=0.68)
    valid, retry_reason = validate_poem(retry_poem, req.experience)
    log_event(request_id, f"prompt retry valid={valid} reason={retry_reason} preview={retry_poem[:140]!r}")
    if valid:
        return retry_poem, "retry_low_temperature", retry_reason
    return FAILURE_MESSAGE, "failed_validation", retry_reason


def keyword_token_ids(keywords: List[str]) -> List[int]:
    ids = set()
    for keyword in keywords:
        for variant in [keyword, " " + keyword, "\n" + keyword]:
            ids.update(tokenizer.encode(variant, add_special_tokens=False))
    return list(ids)


def apply_dada_postprocess(poem: str, dada_intensity: int) -> str:
    if dada_intensity < 70:
        return poem
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    if len(lines) < 3:
        return poem
    if dada_intensity >= 90:
        fractured = []
        for index, line in enumerate(lines):
            fractured.append(line)
            words = line.split()
            if index % 2 == 0 and len(words) >= 2:
                fractured.append(f"{words[0]} / {words[-1]}")
        return "\n".join(fractured[:12])
    lines.insert(2, lines[1])
    return "\n".join(lines[:10])


def experiment_params(req: GenerateRequest) -> Dict[str, float | int | str]:
    if req.languageJump < 45:
        return {"strategy": "guided_top_p", "remove_top_n": 0, "band_size": 70, "topk": min(90, VOCAB_SIZE), "temperature": 0.46 + req.languageJump * 0.003 + req.dadaIntensity * 0.0008, "top_p": 0.78}
    if req.languageJump < 75:
        return {"strategy": "mild_anti_greedy", "remove_top_n": int(1 + req.languageJump * 0.05), "band_size": 85, "topk": min(130, VOCAB_SIZE), "temperature": 0.54 + req.languageJump * 0.004 + req.dadaIntensity * 0.001, "top_p": 0.84}
    return {"strategy": "strong_anti_greedy", "remove_top_n": int(3 + req.languageJump * 0.14), "band_size": 120, "topk": min(EXPERIMENT_TOPK, VOCAB_SIZE), "temperature": 0.68 + req.languageJump * 0.006 + req.dadaIntensity * 0.001, "top_p": 0.9}


def sample_from_candidates(logits: torch.Tensor, candidate_indices: torch.Tensor, top_p: float) -> torch.Tensor:
    candidate_logits = torch.gather(logits, 1, candidate_indices)
    sorted_logits, sorted_pos = torch.sort(candidate_logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    keep = cumulative <= top_p
    keep[:, 0] = True
    filtered_logits = sorted_logits.masked_fill(~keep, -float("inf"))
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    sampled_sorted_pos = torch.multinomial(filtered_probs, num_samples=1)
    sampled_candidate_pos = torch.gather(sorted_pos, 1, sampled_sorted_pos)
    return torch.gather(candidate_indices, 1, sampled_candidate_pos)


def generate_experiment_once(req: GenerateRequest, temperature_multiplier: float = 1.0) -> str:
    prompt = build_experiment_seed_prompt(req)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    prompt_length = input_ids.shape[-1]
    keywords = extract_keywords(req.experience)
    boost_ids = keyword_token_ids(keywords)
    boost_value = 0.8 + req.experienceDensity * 0.025
    params = experiment_params(req)
    remove_top_n = int(params["remove_top_n"])
    band_size = int(params["band_size"])
    topk = int(params["topk"])
    temperature = float(params["temperature"]) * temperature_multiplier
    top_p = float(params["top_p"])
    generated = input_ids
    with torch.no_grad():
        for _ in range(EXPERIMENT_MAX_NEW_TOKENS):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :].float()
            if boost_ids:
                valid_ids = [token_id for token_id in boost_ids if token_id < logits.shape[-1]]
                logits[:, valid_ids] += boost_value
            if tokenizer.eos_token_id is not None:
                logits[:, tokenizer.eos_token_id] -= 0.3
            logits = logits / max(temperature, 0.1)
            _, topk_indices = torch.topk(logits, k=topk, dim=-1)
            start = min(remove_top_n, topk_indices.shape[-1] - 2)
            end = min(start + band_size, topk_indices.shape[-1])
            candidate_indices = topk_indices[:, start:end]
            next_token = sample_from_candidates(logits, candidate_indices, top_p=top_p)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            decoded_so_far = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
            if decoded_so_far.count("\n") >= 7:
                break
    decoded = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
    return apply_dada_postprocess(clean_text(decoded), req.dadaIntensity)


def generate_experiment_mode(req: GenerateRequest, request_id: str) -> Tuple[str, str, str]:
    first_poem = generate_experiment_once(req, temperature_multiplier=1.0)
    valid, reason = validate_poem(first_poem, req.experience)
    log_event(request_id, f"experiment first valid={valid} reason={reason} preview={first_poem[:140]!r}")
    if valid:
        return first_poem, "first_pass_custom_decoding", reason
    retry_poem = generate_experiment_once(req, temperature_multiplier=0.72)
    valid, retry_reason = validate_poem(retry_poem, req.experience)
    log_event(request_id, f"experiment retry valid={valid} reason={retry_reason} preview={retry_poem[:140]!r}")
    if valid:
        return retry_poem, "retry_lower_temperature_custom_decoding", retry_reason
    return FAILURE_MESSAGE, "failed_validation", retry_reason


@app.get("/")
def root():
    return {"status": "ok", "message": "Poem Generator Backend is running.", "model": MODEL_ID, "endpoint": "/generate", "app_version": APP_VERSION, "template_fallback": False}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    keywords = extract_keywords(req.experience)
    exp_params = experiment_params(req)
    log_event(request_id, f"start version={APP_VERSION} mode={req.mode} density={req.experienceDensity} jump={req.languageJump} dada={req.dadaIntensity} keywords={keywords}")
    if req.mode == "experiment":
        poem, validation_status, validation_reason = generate_experiment_mode(req, request_id)
        params = {"app_version": APP_VERSION, "request_id": request_id, "mode": "custom_decoding", "validation_status": validation_status, "validation_reason": validation_reason, "strategy": exp_params["strategy"], "experience_boost": round(0.8 + req.experienceDensity * 0.025, 3), "remove_top_n": exp_params["remove_top_n"], "band_size": exp_params["band_size"], "topk": exp_params["topk"], "top_p": exp_params["top_p"], "max_new_tokens": EXPERIMENT_MAX_NEW_TOKENS, "newline_stop": 7, "temperature": round(float(exp_params["temperature"]), 3)}
    else:
        poem, validation_status, validation_reason = generate_prompt_mode(req, request_id)
        params = {"app_version": APP_VERSION, "request_id": request_id, "mode": "prompt_instruction", "validation_status": validation_status, "validation_reason": validation_reason, "temperature": round(min(0.82, 0.48 + req.languageJump * 0.004), 3), "retry_temperature": 0.42, "top_p": 0.82, "retry_top_p": 0.68, "max_new_tokens": MAX_NEW_TOKENS}
    params["elapsed_seconds"] = round(time.time() - start_time, 3)
    log_event(request_id, f"done status={params['validation_status']} reason={params['validation_reason']} elapsed={params['elapsed_seconds']}s")
    return GenerateResponse(poem=poem, mode=req.mode, model=MODEL_ID, params=params)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
