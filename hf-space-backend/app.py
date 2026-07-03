import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # peft is only required when USE_ADAPTER=true or fallback is used.
    PeftModel = None

APP_VERSION = "finetuned-space-backend-2026-07-03-stage10-v1"

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_ID = os.getenv("MODEL_ID", "z-unghyun/poem-generator-merged")
ADAPTER_ID = os.getenv("ADAPTER_ID", "z-unghyun/poem-generator-lora-adapter")
USE_ADAPTER = os.getenv("USE_ADAPTER", "false").lower() in {"1", "true", "yes", "y"}
ENABLE_ADAPTER_FALLBACK = os.getenv("ENABLE_ADAPTER_FALLBACK", "true").lower() in {"1", "true", "yes", "y"}

SHOW_INVALID_OUTPUTS = True
MODE = "finetuned_experiment"
NEWLINE_STOP = int(os.getenv("NEWLINE_STOP", "3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "60"))
MIN_LAST_LINE_CHARS = int(os.getenv("MIN_LAST_LINE_CHARS", "4"))

SYSTEM_PROMPT = (
    "너는 경험을 3줄 한국어 시로 바꾸는 시 생성 모델이다. "
    "설명 없이 시만 정확히 3줄로 쓴다."
)

STOP_STRINGS = ["<|im_end|>", "<|endoftext|>"]
PROMPT_MARKERS = (
    "경험:",
    "시:",
    "제목:",
    "해설:",
    "assistant",
    "system",
    "user",
    "<|im_start|>",
    "<|im_end|>",
)
PROSE_PATTERNS = (
    "입니다",
    "합니다",
    "하였다",
    "했다",
    "것이다",
    "수 있다",
    "때문이다",
    "이 시는",
    "이 문장은",
    "설명",
    "요약",
    "번역하면",
    "변형한",
    "보여주기",
    "바탕으로",
)
SUSPICIOUS_WORDS = (
    "꾸방",
    "오버보아",
    "보여주기한다",
    "망갈",
    "번역하면",
)
STOPWORDS = {
    "하다",
    "하다가",
    "있다",
    "없는",
    "같은",
    "그리고",
    "에서",
    "으로",
    "에게",
    "끝나고",
    "기다림",
    "갑자기",
    "혼자",
    "새벽에",
    "카페에서",
    "프로젝트를",
    "과제를",
}

app = FastAPI(title="Dadaism Poem Generator Backend", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    experience: str = Field(default="", description="사용자 경험 입력")
    languageJump: int = Field(default=65, ge=0, le=100, description="언어 도약도 0~100")

    class Config:
        extra = "ignore"


class GenerateResponse(BaseModel):
    poem: str
    mode: Literal["finetuned_experiment"]
    model: str
    params: Dict[str, Any]
    # Backward-compatible top-level fields. Frontend can read either top-level or params.*.
    validation_status: Literal["valid", "invalid_shown"]
    validation_reason: str


def env_device_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
    else:
        kwargs.update({"torch_dtype": torch.float32})
    return kwargs


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_merged_model() -> Tuple[Any, Any, torch.device, str, str]:
    print(f"[startup] loading merged model: {MODEL_ID}", flush=True)
    tokenizer = load_tokenizer(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **env_device_kwargs())
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device, MODEL_ID, "merged"


def load_adapter_model() -> Tuple[Any, Any, torch.device, str, str]:
    if PeftModel is None:
        raise RuntimeError("peft is not installed, so adapter loading is unavailable.")

    print(f"[startup] loading base model: {BASE_MODEL_ID}", flush=True)
    print(f"[startup] loading LoRA adapter: {ADAPTER_ID}", flush=True)
    tokenizer = load_tokenizer(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **env_device_kwargs())
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device, ADAPTER_ID, "adapter"


def load_model_stack() -> Tuple[Any, Any, torch.device, str, str]:
    print(
        f"[startup] app_version={APP_VERSION} use_adapter={USE_ADAPTER} "
        f"model_id={MODEL_ID} adapter_id={ADAPTER_ID}",
        flush=True,
    )

    if USE_ADAPTER:
        return load_adapter_model()

    try:
        return load_merged_model()
    except Exception as exc:
        print(f"[startup] merged model load failed: {repr(exc)}", flush=True)
        if not ENABLE_ADAPTER_FALLBACK:
            raise
        print("[startup] falling back to adapter mode", flush=True)
        return load_adapter_model()


TOKENIZER, MODEL, DEVICE, LOADED_MODEL_ID, MODEL_LOAD_MODE = load_model_stack()
print(
    f"[startup] loaded mode={MODEL_LOAD_MODE} model={LOADED_MODEL_ID} "
    f"device={DEVICE} cuda={torch.cuda.is_available()}",
    flush=True,
)


def clamp_int(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(value)))


def build_generation_params(language_jump: int) -> Dict[str, Any]:
    jump = clamp_int(language_jump)

    if jump <= 30:
        strategy = "stable_guided_sampling"
        temperature = 0.42 + (jump / 30.0) * 0.13  # 0.42~0.55
        top_p = 0.82 + (jump / 30.0) * 0.04  # 0.82~0.86
        topk = 45 + int(jump * 0.8)
        remove_top_n = 0
        band_size = 10 + int(jump * 0.2)
    elif jump <= 70:
        strategy = "mild_anti_greedy"
        ratio = (jump - 30) / 40.0
        temperature = 0.55 + ratio * 0.20  # 0.55~0.75
        top_p = 0.86 + ratio * 0.06  # 0.86~0.92
        topk = 70 + int(ratio * 70)
        remove_top_n = 1 if jump >= 45 else 0
        band_size = 16 + int(ratio * 20)
    else:
        strategy = "strong_anti_greedy"
        ratio = (jump - 70) / 30.0
        temperature = 0.75 + ratio * 0.30  # 0.75~1.05
        top_p = 0.92 + ratio * 0.05  # 0.92~0.97
        topk = 140 + int(ratio * 90)
        remove_top_n = 2 + int(ratio * 3)
        band_size = 36 + int(ratio * 24)

    return {
        "language_jump": jump,
        "strategy": strategy,
        "temperature": round(float(temperature), 3),
        "top_p": round(float(top_p), 3),
        "topk": int(topk),
        "remove_top_n": int(remove_top_n),
        "band_size": int(band_size),
        "newline_stop": NEWLINE_STOP,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def build_prompt(experience: str) -> str:
    clean_experience = (experience or "").strip() or "비어 있는 경험"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"경험: {clean_experience}"},
    ]
    if hasattr(TOKENIZER, "apply_chat_template"):
        return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"system: {SYSTEM_PROMPT}\nuser: 경험: {clean_experience}\nassistant:\n"


def apply_custom_logits_filter(logits: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    filtered = logits.float().clone()
    temperature = max(float(params["temperature"]), 1e-5)
    filtered = filtered / temperature

    vocab_size = filtered.shape[-1]
    sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)

    remove_top_n = max(0, min(int(params["remove_top_n"]), vocab_size - 2))
    topk = max(1, min(int(params["topk"]), vocab_size - remove_top_n))
    band_size = max(1, min(int(params["band_size"]), vocab_size - remove_top_n))
    keep_count = max(topk, min(vocab_size - remove_top_n, topk + band_size))

    allowed = torch.full_like(filtered, float("-inf"))
    keep_indices = sorted_indices[:, remove_top_n : remove_top_n + keep_count]
    keep_logits = sorted_logits[:, remove_top_n : remove_top_n + keep_count]
    allowed.scatter_(dim=-1, index=keep_indices, src=keep_logits)

    top_p = float(params["top_p"])
    if top_p < 1.0:
        sorted_allowed, sorted_allowed_indices = torch.sort(allowed, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_allowed, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        allowed.scatter_(
            dim=-1,
            index=sorted_allowed_indices,
            src=sorted_allowed.masked_fill(remove_mask, float("-inf")),
        )

    return allowed


def nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def should_stop_generation(candidate_text: str, token_text: str) -> bool:
    lines = nonempty_lines(candidate_text)
    if len(lines) < NEWLINE_STOP:
        return False

    # Stop when the model tries to start a fourth non-empty line. The caller does not append that token.
    if len(lines) > NEWLINE_STOP:
        return True

    last_line = lines[-1] if lines else ""
    if len(last_line.replace(" ", "")) < MIN_LAST_LINE_CHARS:
        return False

    # Prefer natural line boundary, but also stop before overly long third line on CPU Spaces.
    if "\n" in token_text:
        return True
    if len(last_line) >= 22:
        return True
    if last_line.endswith((".", "?", "!", "…")) and len(last_line) >= 6:
        return True
    return False


def sample_next_token(filtered_logits: torch.Tensor, raw_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(filtered_logits, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum().item() <= 0:
        return torch.argmax(raw_logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate_with_custom_decoding(experience: str, params: Dict[str, Any]) -> str:
    prompt = build_prompt(experience)
    encoded = TOKENIZER(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    generated_ids: List[int] = []
    past_key_values = None
    next_input_ids = input_ids

    for step in range(int(params["max_new_tokens"])):
        if past_key_values is None:
            outputs = MODEL(input_ids=next_input_ids, attention_mask=attention_mask, use_cache=True)
        else:
            outputs = MODEL(input_ids=next_input_ids, past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        raw_logits = outputs.logits[:, -1, :]
        filtered_logits = apply_custom_logits_filter(raw_logits, params)
        next_token = sample_next_token(filtered_logits, raw_logits).to(DEVICE)
        token_id = int(next_token.item())

        if token_id == TOKENIZER.eos_token_id:
            break

        token_text = TOKENIZER.decode([token_id], skip_special_tokens=False)
        candidate_ids = generated_ids + [token_id]
        candidate_text = TOKENIZER.decode(candidate_ids, skip_special_tokens=True)

        if any(stop in candidate_text for stop in STOP_STRINGS):
            break

        if should_stop_generation(candidate_text, token_text):
            # If the token only completes the third line, keep it. If it starts a fourth line, do not keep it.
            if len(nonempty_lines(candidate_text)) <= NEWLINE_STOP:
                generated_ids.append(token_id)
            break

        generated_ids.append(token_id)
        next_input_ids = next_token

    return TOKENIZER.decode(generated_ids, skip_special_tokens=True)


def clean_generated_text(text: str) -> str:
    # Do not rewrite, template, or fallback. Only remove transport/special-token residue.
    cleaned = text or ""
    for stop in STOP_STRINGS:
        cleaned = cleaned.replace(stop, "")
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def normalize_for_repetition(line: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"[^A-Za-z가-힣0-9]", "", line)).lower()


def has_repetitive_lines(lines: List[str]) -> bool:
    normalized = [normalize_for_repetition(line) for line in lines if normalize_for_repetition(line)]
    if len(normalized) != len(set(normalized)):
        return True
    if len(normalized) >= 3 and len(set(normalized)) <= 1:
        return True
    return False


def hangul_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z가-힣一-龥]", text)
    if not letters:
        return 0.0
    hangul = re.findall(r"[가-힣]", text)
    return len(hangul) / max(len(letters), 1)


def looks_like_broken_korean(poem: str) -> bool:
    compact = re.sub(r"\s+", "", poem)
    if not compact:
        return True
    if len(re.findall(r"[\u4e00-\u9fff]", poem)) >= 5:
        return True
    if len(compact) >= 20 and hangul_ratio(poem) < 0.45:
        return True
    if re.search(r"[A-Za-z]{12,}", poem):
        return True
    if re.search(r"(.)\1{7,}", compact):
        return True
    return False


def has_prompt_marker_leak(raw_output: str, poem: str) -> bool:
    check = f"{raw_output}\n{poem}".lower()
    return any(marker.lower() in check for marker in PROMPT_MARKERS)


def has_list_like_output(lines: List[str]) -> bool:
    count = 0
    for line in lines:
        if re.match(r"^\s*[-*•]?\s*\d+[\.)]\s+", line):
            count += 1
        elif re.match(r"^\s*[-*•]\s+", line):
            count += 1
    return count >= 2


def has_suspicious_random_word(poem: str) -> bool:
    if any(word in poem for word in SUSPICIOUS_WORDS):
        return True
    # Odd non-Korean fragments inside mostly Korean output.
    if re.search(r"[A-Za-z]{6,}", poem) and "AI" not in poem:
        return True
    # Jamo fragments usually mean token-level collapse or typo-like output.
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ]{2,}", poem):
        return True
    return False


def has_unfinished_sentence(lines: List[str]) -> bool:
    unfinished_patterns = (
        r"먹으$",
        r"하$",
        r"되$",
        r"임$",
        r"이며$",
        r"면서$",
        r"다가$",
        r"으로$",
        r"에게$",
    )
    return any(any(re.search(pattern, line.strip()) for pattern in unfinished_patterns) for line in lines)


def normalize_keyword(token: str) -> str:
    token = re.sub(r"[^A-Za-z가-힣0-9]", "", token).strip()
    suffixes = (
        "에서",
        "에게",
        "으로",
        "처럼",
        "까지",
        "부터",
        "하고",
        "하며",
        "하면",
        "해서",
        "하다가",
        "하며",
        "이며",
        "이랑",
        "랑",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "도",
        "만",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
                changed = True
                break
    return token


def extract_experience_keywords(experience: str, limit: int = 10) -> List[str]:
    raw_tokens = re.split(r"\s+|,|\.|/|·", experience or "")
    keywords: List[str] = []
    for raw in raw_tokens:
        token = normalize_keyword(raw)
        if not token or token in STOPWORDS:
            continue
        if len(token) < 2 and not re.match(r"[A-Za-z0-9]{2,}", token):
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords[:limit]


def has_experience_keyword_match(experience: str, poem: str) -> Tuple[bool, List[str]]:
    keywords = extract_experience_keywords(experience)
    if not keywords:
        return True, []
    compact_poem = re.sub(r"\s+", "", poem)
    matched = [kw for kw in keywords if kw.replace(" ", "") in compact_poem]
    return len(matched) > 0, matched


def validate_poem(raw_output: str, poem: str, experience: str) -> Tuple[str, str, Dict[str, Any]]:
    lines = nonempty_lines(poem)
    diagnostics: Dict[str, Any] = {
        "line_count": len(lines),
        "lines": lines,
        "experience_keywords": extract_experience_keywords(experience),
        "matched_keywords": [],
    }

    if not poem.strip():
        return "invalid_shown", "empty_output", diagnostics

    if has_prompt_marker_leak(raw_output, poem):
        return "invalid_shown", "prompt_marker_leak", diagnostics

    if len(lines) != 3:
        return "invalid_shown", f"not_three_lines:{len(lines)}", diagnostics

    if has_list_like_output(lines):
        return "invalid_shown", "list_like_output", diagnostics

    if any(len(line) > 70 for line in lines):
        return "invalid_shown", "line_too_long", diagnostics

    if has_repetitive_lines(lines):
        return "invalid_shown", "repetitive_lines", diagnostics

    if any(any(pattern in line for pattern in PROSE_PATTERNS) for line in lines):
        return "invalid_shown", "prose_report_like_output", diagnostics

    if looks_like_broken_korean(poem):
        return "invalid_shown", "broken_korean_like_output", diagnostics

    if has_suspicious_random_word(poem):
        return "invalid_shown", "suspicious_random_word", diagnostics

    if has_unfinished_sentence(lines):
        return "invalid_shown", "unfinished_sentence", diagnostics

    has_match, matched_keywords = has_experience_keyword_match(experience, poem)
    diagnostics["matched_keywords"] = matched_keywords
    if not has_match:
        return "invalid_shown", "no_experience_keyword_match", diagnostics

    return "valid", "ok", diagnostics


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app_version": APP_VERSION,
        "mode": MODE,
        "model": LOADED_MODEL_ID,
        "model_load_mode": MODEL_LOAD_MODE,
        "model_id_env": MODEL_ID,
        "base_model": BASE_MODEL_ID,
        "adapter_id": ADAPTER_ID,
        "use_adapter": USE_ADAPTER,
        "enable_adapter_fallback": ENABLE_ADAPTER_FALLBACK,
        "show_invalid_outputs": SHOW_INVALID_OUTPUTS,
        "newline_stop": NEWLINE_STOP,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    started = time.time()
    request_id = str(uuid.uuid4())
    experience = (req.experience or "").strip()
    params = build_generation_params(req.languageJump)

    raw_output = generate_with_custom_decoding(experience, params)
    poem = clean_generated_text(raw_output)
    validation_status, validation_reason, diagnostics = validate_poem(raw_output, poem, experience)
    elapsed_seconds = round(time.time() - started, 3)

    params.update(
        {
            "app_version": APP_VERSION,
            "request_id": request_id,
            "elapsed_seconds": elapsed_seconds,
            "validation_status": validation_status,
            "validation_reason": validation_reason,
            "model_load_mode": MODEL_LOAD_MODE,
            "model": LOADED_MODEL_ID,
            "show_invalid_outputs": SHOW_INVALID_OUTPUTS,
            "line_count": diagnostics.get("line_count"),
            "experience_keywords": diagnostics.get("experience_keywords", []),
            "matched_keywords": diagnostics.get("matched_keywords", []),
        }
    )

    print(
        f"[generate][{request_id}] status={validation_status} reason={validation_reason} "
        f"jump={params['language_jump']} elapsed={elapsed_seconds}s",
        flush=True,
    )

    return GenerateResponse(
        poem=poem,
        mode=MODE,
        model=LOADED_MODEL_ID,
        params=params,
        validation_status=validation_status,
        validation_reason=validation_reason,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
