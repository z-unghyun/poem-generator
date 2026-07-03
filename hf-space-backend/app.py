import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Tuple

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # peft is only required when USE_ADAPTER=true or fallback is used.
    PeftModel = None

APP_VERSION = "finetuned-space-backend-2026-07-03-stage16-decode-guard-v1"

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_ID = os.getenv("MODEL_ID", "z-unghyun/poem-generator-merged")
ADAPTER_ID = os.getenv("ADAPTER_ID", "z-unghyun/poem-generator-lora-adapter")
USE_ADAPTER = os.getenv("USE_ADAPTER", "false").lower() in {"1", "true", "yes", "y"}
ENABLE_ADAPTER_FALLBACK = os.getenv("ENABLE_ADAPTER_FALLBACK", "true").lower() in {"1", "true", "yes", "y"}
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "short").lower()

SHOW_INVALID_OUTPUTS = True
MODE = "finetuned_experiment"
NEWLINE_STOP = int(os.getenv("NEWLINE_STOP", "3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "52"))
MIN_LAST_LINE_CHARS = int(os.getenv("MIN_LAST_LINE_CHARS", "4"))
THIRD_LINE_MAX_CHARS = int(os.getenv("THIRD_LINE_MAX_CHARS", "22"))
CPU_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "2"))

ENABLE_EXPERIMENT_LOGS = os.getenv("ENABLE_EXPERIMENT_LOGS", "true").lower() in {"1", "true", "yes", "y"}
LOG_EXPERIENCE = os.getenv("LOG_EXPERIENCE", "true").lower() in {"1", "true", "yes", "y"}
LOG_POEM = os.getenv("LOG_POEM", "true").lower() in {"1", "true", "yes", "y"}
EXPERIMENT_LOG_PATH = os.getenv("EXPERIMENT_LOG_PATH", "/tmp/poem-generator/logs.jsonl")
EXPOSE_EXPERIMENT_LOGS = os.getenv("EXPOSE_EXPERIMENT_LOGS", "false").lower() in {"1", "true", "yes", "y"}
LOG_READ_TOKEN = os.getenv("LOG_READ_TOKEN", "").strip()
LOG_MAX_READ_LINES = int(os.getenv("LOG_MAX_READ_LINES", "500"))

ENABLE_EOJEOL_GUARD = os.getenv("ENABLE_EOJEOL_GUARD", "true").lower() in {"1", "true", "yes", "y"}
ENABLE_BAD_TOKEN_GUARD = os.getenv("ENABLE_BAD_TOKEN_GUARD", "true").lower() in {"1", "true", "yes", "y"}
ENABLE_REPETITION_PENALTY = os.getenv("ENABLE_REPETITION_PENALTY", "true").lower() in {"1", "true", "yes", "y"}
ENABLE_NEWLINE_BIAS = os.getenv("ENABLE_NEWLINE_BIAS", "true").lower() in {"1", "true", "yes", "y"}
BAD_TOKEN_SCAN_TOP_N = int(os.getenv("BAD_TOKEN_SCAN_TOP_N", "48"))
NEWLINE_MIN_CHARS = int(os.getenv("NEWLINE_MIN_CHARS", "6"))
NEWLINE_BIAS_START_CHARS = int(os.getenv("NEWLINE_BIAS_START_CHARS", "10"))
NEWLINE_BIAS_VALUE = float(os.getenv("NEWLINE_BIAS_VALUE", "1.15"))
NEWLINE_SUPPRESS_VALUE = float(os.getenv("NEWLINE_SUPPRESS_VALUE", "6.0"))
REPETITION_PENALTY_LOW = float(os.getenv("REPETITION_PENALTY_LOW", "1.20"))
REPETITION_PENALTY_MID = float(os.getenv("REPETITION_PENALTY_MID", "1.12"))
REPETITION_PENALTY_HIGH = float(os.getenv("REPETITION_PENALTY_HIGH", "1.05"))
EOJEOL_TEMPERATURE = float(os.getenv("EOJEOL_TEMPERATURE", "0.45"))
EOJEOL_TOP_P = float(os.getenv("EOJEOL_TOP_P", "0.82"))
EOJEOL_TOPK = int(os.getenv("EOJEOL_TOPK", "35"))
EOJEOL_BAND_SIZE = int(os.getenv("EOJEOL_BAND_SIZE", "8"))

SYSTEM_PROMPT = "너는 경험을 3줄 한국어 시로 바꾸는 시 생성 모델이다. 설명 없이 시만 쓴다."
SHORT_PROMPT_TEMPLATE = "경험을 3줄 시로.\n경험: {experience}\n시:\n"

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
BAD_TEXT_FRAGMENTS = (
    "Answer",
    "answer",
    "Pros",
    "bytes",
    "String",
    "speedsyn",
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

if not torch.cuda.is_available():
    try:
        torch.set_num_threads(max(1, CPU_NUM_THREADS))
    except Exception as exc:
        print(f"[startup] torch.set_num_threads skipped: {repr(exc)}", flush=True)

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


def finalize_model(model: Any) -> Any:
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model


def load_merged_model() -> Tuple[Any, Any, torch.device, str, str]:
    print(f"[startup] loading merged model: {MODEL_ID}", flush=True)
    tokenizer = load_tokenizer(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **env_device_kwargs())
    model = finalize_model(model)
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
    model = finalize_model(model)
    device = next(model.parameters()).device
    return tokenizer, model, device, ADAPTER_ID, "adapter"


def load_model_stack() -> Tuple[Any, Any, torch.device, str, str]:
    print(
        f"[startup] app_version={APP_VERSION} use_adapter={USE_ADAPTER} "
        f"model_id={MODEL_ID} adapter_id={ADAPTER_ID} prompt_style={PROMPT_STYLE}",
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
NEWLINE_TOKEN_IDS = sorted(
    set(
        token_id
        for text in ("\n", "\n\n")
        for token_id in TOKENIZER.encode(text, add_special_tokens=False)
    )
)
print(
    f"[startup] loaded mode={MODEL_LOAD_MODE} model={LOADED_MODEL_ID} "
    f"device={DEVICE} cuda={torch.cuda.is_available()} newline_token_ids={NEWLINE_TOKEN_IDS}",
    flush=True,
)


def clamp_int(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(value)))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_generation_params(language_jump: int) -> Dict[str, Any]:
    jump = clamp_int(language_jump)

    if jump <= 30:
        strategy = "stable_guided_sampling"
        temperature = 0.40 + (jump / 30.0) * 0.12  # 0.40~0.52
        top_p = 0.80 + (jump / 30.0) * 0.04  # 0.80~0.84
        topk = 38 + int(jump * 0.7)
        remove_top_n = 0
        band_size = 8 + int(jump * 0.18)
    elif jump <= 70:
        strategy = "mild_anti_greedy"
        ratio = (jump - 30) / 40.0
        temperature = 0.52 + ratio * 0.18  # 0.52~0.70
        top_p = 0.84 + ratio * 0.06  # 0.84~0.90
        topk = 58 + int(ratio * 56)
        remove_top_n = 1 if jump >= 48 else 0
        band_size = 12 + int(ratio * 16)
    else:
        strategy = "strong_anti_greedy_guarded"
        ratio = (jump - 70) / 30.0
        temperature = 0.70 + ratio * 0.12  # 0.70~0.82
        top_p = 0.88 + ratio * 0.02  # 0.88~0.90
        topk = 96 + int(ratio * 14)  # 96~110
        remove_top_n = 1
        band_size = 18 + int(ratio * 4)  # 18~22

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
        "prompt_style": PROMPT_STYLE,
        "use_kv_cache": True,
        "eojeol_guard": ENABLE_EOJEOL_GUARD,
        "bad_token_guard": ENABLE_BAD_TOKEN_GUARD,
        "repetition_penalty_enabled": ENABLE_REPETITION_PENALTY,
        "newline_bias_enabled": ENABLE_NEWLINE_BIAS,
    }


def build_prompt(experience: str) -> str:
    clean_experience = (experience or "").strip() or "비어 있는 경험"

    if PROMPT_STYLE == "chat":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"경험: {clean_experience}"},
        ]
        if hasattr(TOKENIZER, "apply_chat_template"):
            return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"system: {SYSTEM_PROMPT}\nuser: 경험: {clean_experience}\nassistant:\n"

    return SHORT_PROMPT_TEMPLATE.format(experience=clean_experience)


def current_line_info(text: str) -> Tuple[int, str, int]:
    lines = nonempty_lines(text)
    line_count = len(lines)
    last_line = lines[-1] if lines else ""
    last_line_chars = len(last_line.replace(" ", ""))
    return line_count, last_line, last_line_chars


def is_inside_eojeol(generated_text: str) -> bool:
    if not generated_text:
        return False
    last = generated_text[-1]
    if last.isspace():
        return False
    if last in {".", ",", "?", "!", "…", ";", ":", "'", '"', "“", "”", "‘", "’", "(", ")", "[", "]", "{", "}", "-", "—"}:
        return False
    return bool(re.match(r"[가-힣A-Za-z0-9]", last))


def get_effective_params(params: Dict[str, Any], generated_text: str) -> Tuple[Dict[str, Any], bool]:
    if not ENABLE_EOJEOL_GUARD or not is_inside_eojeol(generated_text):
        return params, False

    stable = dict(params)
    stable["temperature"] = min(float(params["temperature"]), EOJEOL_TEMPERATURE)
    stable["top_p"] = min(float(params["top_p"]), EOJEOL_TOP_P)
    stable["topk"] = min(int(params["topk"]), EOJEOL_TOPK)
    stable["remove_top_n"] = 0
    stable["band_size"] = min(int(params["band_size"]), EOJEOL_BAND_SIZE)
    stable["strategy"] = f"{params.get('strategy', 'sampling')}_inside_eojeol_stable"
    return stable, True


def get_repetition_penalty(language_jump: int) -> float:
    if language_jump <= 25:
        return REPETITION_PENALTY_LOW
    if language_jump <= 70:
        return REPETITION_PENALTY_MID
    return REPETITION_PENALTY_HIGH


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int], params: Dict[str, Any]) -> torch.Tensor:
    if not ENABLE_REPETITION_PENALTY or not generated_ids:
        return logits
    penalty = max(1.0, get_repetition_penalty(int(params.get("language_jump", 65))))
    if penalty <= 1.0:
        return logits
    for token_id in set(generated_ids):
        if 0 <= token_id < logits.shape[-1]:
            logits[:, token_id] = logits[:, token_id] / penalty
    return logits


def apply_newline_bias(logits: torch.Tensor, generated_text: str) -> Tuple[torch.Tensor, str]:
    if not ENABLE_NEWLINE_BIAS or not NEWLINE_TOKEN_IDS:
        return logits, "none"

    line_count, _, last_line_chars = current_line_info(generated_text)
    valid_ids = [token_id for token_id in NEWLINE_TOKEN_IDS if 0 <= token_id < logits.shape[-1]]
    if not valid_ids:
        return logits, "none"

    if line_count >= NEWLINE_STOP:
        logits[:, valid_ids] = logits[:, valid_ids] - NEWLINE_SUPPRESS_VALUE
        return logits, "suppress_after_three_lines"

    if last_line_chars < NEWLINE_MIN_CHARS:
        logits[:, valid_ids] = logits[:, valid_ids] - NEWLINE_SUPPRESS_VALUE
        return logits, "suppress_too_short_line"

    if line_count in {1, 2} and last_line_chars >= NEWLINE_BIAS_START_CHARS:
        logits[:, valid_ids] = logits[:, valid_ids] + NEWLINE_BIAS_VALUE
        return logits, "boost_line_break"

    return logits, "none"


def is_bad_token_text(token_text: str) -> bool:
    if not token_text:
        return False
    if "�" in token_text:
        return True
    if any(fragment in token_text for fragment in BAD_TEXT_FRAGMENTS):
        return True
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ]{2,}", token_text):
        return True
    if re.search(r"[A-Za-z]{4,}", token_text) and "AI" not in token_text:
        return True
    if len(re.findall(r"[\u4e00-\u9fff]", token_text)) >= 2:
        return True
    return False


def apply_bad_token_guard(logits: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    if not ENABLE_BAD_TOKEN_GUARD:
        return logits
    scan_n = max(1, min(BAD_TOKEN_SCAN_TOP_N, logits.shape[-1]))
    _, candidate_ids = torch.topk(logits, k=scan_n, dim=-1, largest=True, sorted=False)
    blocked = 0
    for token_id in candidate_ids[0].tolist():
        token_text = TOKENIZER.decode([int(token_id)], skip_special_tokens=False)
        if is_bad_token_text(token_text):
            logits[:, int(token_id)] = float("-inf")
            blocked += 1
    stats["bad_tokens_blocked"] = stats.get("bad_tokens_blocked", 0) + blocked
    return logits


def apply_custom_logits_filter(
    logits: torch.Tensor,
    params: Dict[str, Any],
    generated_ids: List[int],
    generated_text: str,
    stats: Dict[str, Any],
) -> torch.Tensor:
    effective_params, eojeol_stabilized = get_effective_params(params, generated_text)
    if eojeol_stabilized:
        stats["eojeol_stabilized_steps"] = stats.get("eojeol_stabilized_steps", 0) + 1

    scaled = logits.float().clone()
    scaled = apply_repetition_penalty(scaled, generated_ids, params)
    scaled, newline_action = apply_newline_bias(scaled, generated_text)
    if newline_action != "none":
        stats[f"newline_{newline_action}_steps"] = stats.get(f"newline_{newline_action}_steps", 0) + 1

    scaled = scaled / max(float(effective_params["temperature"]), 1e-5)
    vocab_size = scaled.shape[-1]

    remove_top_n = max(0, min(int(effective_params["remove_top_n"]), vocab_size - 2))
    topk = max(1, min(int(effective_params["topk"]), vocab_size - remove_top_n))
    band_size = max(1, min(int(effective_params["band_size"]), vocab_size - remove_top_n))
    candidate_count = max(1, min(vocab_size, remove_top_n + topk + band_size))

    top_values, top_indices = torch.topk(scaled, k=candidate_count, dim=-1, largest=True, sorted=True)
    kept_values = top_values[:, remove_top_n:]
    kept_indices = top_indices[:, remove_top_n:]

    if kept_values.numel() == 0:
        return scaled

    top_p = float(effective_params["top_p"])
    if top_p < 1.0:
        kept_probs = torch.softmax(kept_values, dim=-1)
        cumulative_probs = torch.cumsum(kept_probs, dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        kept_values = kept_values.masked_fill(remove_mask, float("-inf"))

    filtered = torch.full_like(scaled, float("-inf"))
    filtered.scatter_(dim=-1, index=kept_indices, src=kept_values)
    filtered = apply_bad_token_guard(filtered, stats)
    return filtered


def nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def should_stop_generation(candidate_text: str, token_text: str) -> Tuple[bool, str]:
    lines = nonempty_lines(candidate_text)
    if len(lines) < NEWLINE_STOP:
        return False, "continue"

    if len(lines) > NEWLINE_STOP:
        return True, "started_extra_line"

    last_line = lines[-1] if lines else ""
    if len(last_line.replace(" ", "")) < MIN_LAST_LINE_CHARS:
        return False, "continue"

    if "\n" in token_text:
        return True, "third_line_newline"
    if len(last_line) >= THIRD_LINE_MAX_CHARS:
        return True, "third_line_max_chars"
    if last_line.endswith((".", "?", "!", "…")) and len(last_line) >= 6:
        return True, "third_line_terminal_punctuation"
    return False, "continue"


def sample_next_token(filtered_logits: torch.Tensor, raw_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(filtered_logits, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum().item() <= 0:
        return torch.argmax(raw_logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate_with_custom_decoding(experience: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    prompt = build_prompt(experience)
    encoded = TOKENIZER(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    generated_ids: List[int] = []
    generated_text = ""
    past_key_values = None
    next_input_ids = input_ids
    stopped_reason = "max_new_tokens"
    decode_steps = 0
    guard_stats: Dict[str, Any] = {
        "bad_tokens_blocked": 0,
        "eojeol_stabilized_steps": 0,
    }
    loop_started = time.time()

    for step in range(int(params["max_new_tokens"])):
        if past_key_values is None:
            outputs = MODEL(input_ids=next_input_ids, attention_mask=attention_mask, use_cache=True)
        else:
            outputs = MODEL(input_ids=next_input_ids, past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        raw_logits = outputs.logits[:, -1, :]
        filtered_logits = apply_custom_logits_filter(raw_logits, params, generated_ids, generated_text, guard_stats)
        next_token = sample_next_token(filtered_logits, raw_logits).to(DEVICE)
        token_id = int(next_token.item())
        decode_steps = step + 1

        if token_id == TOKENIZER.eos_token_id:
            stopped_reason = "eos_token"
            break

        token_text = TOKENIZER.decode([token_id], skip_special_tokens=False)
        candidate_text = generated_text + token_text

        if any(stop in candidate_text for stop in STOP_STRINGS):
            stopped_reason = "stop_string"
            break

        should_stop, reason = should_stop_generation(candidate_text, token_text)
        if should_stop:
            if len(nonempty_lines(candidate_text)) <= NEWLINE_STOP:
                generated_ids.append(token_id)
                generated_text = candidate_text
            stopped_reason = reason
            break

        generated_ids.append(token_id)
        generated_text = candidate_text
        next_input_ids = next_token

    generation_seconds = max(time.time() - loop_started, 1e-6)
    final_text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
    stats = {
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": len(generated_ids),
        "decode_steps": decode_steps,
        "generation_seconds": round(generation_seconds, 3),
        "tokens_per_second": round(len(generated_ids) / generation_seconds, 3),
        "stopped_reason": stopped_reason,
        **guard_stats,
    }
    return final_text, stats


def clean_generated_text(text: str) -> str:
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
    if re.search(r"[A-Za-z]{6,}", poem) and "AI" not in poem:
        return True
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


def build_experiment_log_record(
    *,
    timestamp: str,
    request_id: str,
    experience: str,
    poem: str,
    params: Dict[str, Any],
    validation_status: str,
    validation_reason: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "request_id": request_id,
        "experience": experience if LOG_EXPERIENCE else "[redacted]",
        "poem": poem if LOG_POEM else "[redacted]",
        "languageJump": params.get("language_jump"),
        "validation_status": validation_status,
        "validation_reason": validation_reason,
        "strategy": params.get("strategy"),
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "topk": params.get("topk"),
        "remove_top_n": params.get("remove_top_n"),
        "elapsed_seconds": params.get("elapsed_seconds"),
        "line_count": params.get("line_count"),
        "model_load_mode": params.get("model_load_mode"),
        "model": params.get("model"),
        "app_version": params.get("app_version"),
        "bad_tokens_blocked": params.get("bad_tokens_blocked"),
        "eojeol_stabilized_steps": params.get("eojeol_stabilized_steps"),
    }


def append_experiment_log(record: Dict[str, Any]) -> None:
    """Append one generation result to JSONL and print a minimal backend log line.

    This is intended for development/debug logging. On free Spaces the default path is
    ephemeral, so logs can disappear after restart/rebuild unless a persistent volume is used.
    """
    if not ENABLE_EXPERIMENT_LOGS:
        return

    safe_summary = {
        "timestamp": record.get("timestamp"),
        "request_id": record.get("request_id"),
        "languageJump": record.get("languageJump"),
        "validation_status": record.get("validation_status"),
        "validation_reason": record.get("validation_reason"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "bad_tokens_blocked": record.get("bad_tokens_blocked"),
        "eojeol_stabilized_steps": record.get("eojeol_stabilized_steps"),
    }
    print(f"[experiment_log] {json.dumps(safe_summary, ensure_ascii=False)}", flush=True)

    try:
        log_dir = os.path.dirname(EXPERIMENT_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(EXPERIMENT_LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[experiment_log_error] {repr(exc)}", flush=True)


def require_log_access(token: str) -> None:
    if not EXPOSE_EXPERIMENT_LOGS:
        raise HTTPException(status_code=404, detail="experiment log endpoints are disabled")
    if not LOG_READ_TOKEN:
        raise HTTPException(status_code=403, detail="LOG_READ_TOKEN is not configured")
    if token != LOG_READ_TOKEN:
        raise HTTPException(status_code=403, detail="invalid log token")


def read_recent_log_records(limit: int = 200) -> List[Dict[str, Any]]:
    if not os.path.exists(EXPERIMENT_LOG_PATH):
        return []
    bounded_limit = max(1, min(int(limit), LOG_MAX_READ_LINES))
    records: List[Dict[str, Any]] = []
    with open(EXPERIMENT_LOG_PATH, "r", encoding="utf-8") as fp:
        lines = fp.readlines()[-bounded_limit:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


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
        "prompt_style": PROMPT_STYLE,
        "use_kv_cache": True,
        "torch_num_threads": CPU_NUM_THREADS if not torch.cuda.is_available() else "cuda",
        "experiment_logs_enabled": ENABLE_EXPERIMENT_LOGS,
        "experiment_log_path": EXPERIMENT_LOG_PATH,
        "experiment_log_endpoints_exposed": EXPOSE_EXPERIMENT_LOGS,
        "log_experience": LOG_EXPERIENCE,
        "log_poem": LOG_POEM,
        "decode_guards": {
            "eojeol_guard": ENABLE_EOJEOL_GUARD,
            "bad_token_guard": ENABLE_BAD_TOKEN_GUARD,
            "repetition_penalty": ENABLE_REPETITION_PENALTY,
            "newline_bias": ENABLE_NEWLINE_BIAS,
            "bad_token_scan_top_n": BAD_TOKEN_SCAN_TOP_N,
            "newline_token_ids": NEWLINE_TOKEN_IDS,
        },
    }


@app.get("/logs/summary")
def logs_summary(token: str = Query(default=""), limit: int = Query(default=200, ge=1, le=500)) -> Dict[str, Any]:
    require_log_access(token)
    records = read_recent_log_records(limit=limit)
    reason_counts = Counter(str(record.get("validation_reason", "unknown")) for record in records)
    status_counts = Counter(str(record.get("validation_status", "unknown")) for record in records)
    jump_counts = Counter(str(record.get("languageJump", "unknown")) for record in records)
    return {
        "app_version": APP_VERSION,
        "log_path": EXPERIMENT_LOG_PATH,
        "records_read": len(records),
        "validation_reason_counts": dict(reason_counts),
        "validation_status_counts": dict(status_counts),
        "languageJump_counts": dict(jump_counts),
        "latest_request_ids": [record.get("request_id") for record in records[-20:]],
    }


@app.get("/logs/{request_id}")
def log_by_request_id(request_id: str, token: str = Query(default="")) -> Dict[str, Any]:
    require_log_access(token)
    for record in reversed(read_recent_log_records(limit=LOG_MAX_READ_LINES)):
        if record.get("request_id") == request_id:
            return {"found": True, "record": record}
    return {"found": False, "request_id": request_id}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    started = time.time()
    timestamp = utc_timestamp()
    request_id = str(uuid.uuid4())
    experience = (req.experience or "").strip()
    params = build_generation_params(req.languageJump)

    raw_output, generation_stats = generate_with_custom_decoding(experience, params)
    poem = clean_generated_text(raw_output)
    validation_status, validation_reason, diagnostics = validate_poem(raw_output, poem, experience)
    elapsed_seconds = round(time.time() - started, 3)

    params.update(
        {
            "app_version": APP_VERSION,
            "timestamp": timestamp,
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
            **generation_stats,
        }
    )

    log_record = build_experiment_log_record(
        timestamp=timestamp,
        request_id=request_id,
        experience=experience,
        poem=poem,
        params=params,
        validation_status=validation_status,
        validation_reason=validation_reason,
    )
    append_experiment_log(log_record)

    print(
        f"[generate][{request_id}] status={validation_status} reason={validation_reason} "
        f"jump={params['language_jump']} tokens={params.get('generated_tokens')} "
        f"tok_s={params.get('tokens_per_second')} stopped={params.get('stopped_reason')} "
        f"bad_blocked={params.get('bad_tokens_blocked')} eojeol_steps={params.get('eojeol_stabilized_steps')} "
        f"elapsed={elapsed_seconds}s",
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
