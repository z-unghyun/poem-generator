import os
import re
import time
import uuid
from typing import Any, Dict, List, Literal, Tuple

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_VERSION = "finetuned-space-backend-2026-07-02-1"
MODEL_ID = os.getenv("MODEL_ID", "z-unghyun/poem-generator-merged")
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SHOW_INVALID_OUTPUTS = True
NEWLINE_STOP = 3
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "60"))

SYSTEM_PROMPT = "너는 경험을 3줄 한국어 시로 바꾸는 시 생성 모델이다. 설명 없이 시만 정확히 3줄로 쓴다."
STOP_STRINGS = ["<|im_end|>", "<|endoftext|>"]
PROSE_PATTERNS = ("입니다", "합니다", "것이다", "설명한다", "의미한다", "중요하다", "이 시는", "경험을 바탕으로")
MARKER_PATTERNS = ("제목:", "해설:", "시 본문:", "경험:", "assistant", "system", "user")

app = FastAPI(title="Poem Generator Backend", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    experience: str = ""
    languageJump: int = Field(default=65, ge=0, le=100)
    mode: str | None = None

    class Config:
        extra = "ignore"


class GenerateResponse(BaseModel):
    poem: str
    mode: Literal["finetuned_experiment"]
    model: str
    validation_status: Literal["valid", "invalid_shown"]
    validation_reason: str
    params: Dict[str, Any]


def load_model() -> Tuple[Any, Any, torch.device]:
    print(f"[startup] app_version={APP_VERSION} model={MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs.update({"dtype": torch.float16, "device_map": "auto"})
    else:
        load_kwargs.update({"dtype": torch.float32, "low_cpu_mem_usage": True})

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    device = next(model.parameters()).device
    print(f"[startup] loaded device={device} cuda={torch.cuda.is_available()}", flush=True)
    return tokenizer, model, device


tokenizer, model, DEVICE = load_model()


def clamp_int(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(value)))


def build_generation_params(language_jump: int) -> Dict[str, Any]:
    jump = clamp_int(language_jump)
    remove_top_n = 0
    if jump >= 90:
        remove_top_n = 3
    elif jump >= 70:
        remove_top_n = 2
    elif jump >= 45:
        remove_top_n = 1

    return {
        "language_jump": jump,
        "temperature": round(0.55 + jump * 0.006, 3),
        "top_p": round(min(0.96, 0.82 + jump * 0.0014), 3),
        "top_k": int(35 + jump * 1.45),
        "remove_top_n": remove_top_n,
        "band_size": int(8 + jump * 0.32),
        "newline_stop": NEWLINE_STOP,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def build_prompt(experience: str) -> str:
    experience = experience.strip() or "비어 있는 경험"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"경험: {experience}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"system: {SYSTEM_PROMPT}\nuser: 경험: {experience}\nassistant:\n"


def filter_logits(logits: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    logits = logits.float().clone()
    temperature = max(float(params["temperature"]), 1e-5)
    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    vocab_size = logits.shape[-1]

    remove_top_n = min(int(params["remove_top_n"]), vocab_size - 1)
    top_k = max(1, min(int(params["top_k"]), vocab_size - remove_top_n))
    band_size = max(1, int(params["band_size"]))
    keep_count = max(top_k, min(vocab_size - remove_top_n, top_k + band_size))

    filtered = torch.full_like(logits, float("-inf"))
    keep_indices = sorted_indices[:, remove_top_n : remove_top_n + keep_count]
    keep_logits = sorted_logits[:, remove_top_n : remove_top_n + keep_count]
    filtered.scatter_(dim=-1, index=keep_indices, src=keep_logits)

    top_p = float(params["top_p"])
    if top_p < 1.0:
        sorted_filtered, sorted_filtered_indices = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_filtered, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        indices_to_remove = sorted_filtered_indices[remove_mask]
        filtered[0, indices_to_remove] = float("-inf")

    return filtered


def count_nonempty_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def stop_after_three_lines(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= NEWLINE_STOP and len(lines[-1]) >= 4 and "\n" in text


def generate_with_custom_decoding(experience: str, params: Dict[str, Any]) -> str:
    prompt = build_prompt(experience)
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    generated_ids: List[int] = []
    past_key_values = None
    next_input_ids = input_ids

    with torch.no_grad():
        for step in range(int(params["max_new_tokens"])):
            if past_key_values is None:
                outputs = model(input_ids=next_input_ids, attention_mask=attention_mask, use_cache=True)
            else:
                outputs = model(input_ids=next_input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            filtered_logits = filter_logits(logits, params)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())

            if token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)
            next_input_ids = next_token.to(DEVICE)

            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(stop in decoded for stop in STOP_STRINGS):
                break
            if step >= 10 and stop_after_three_lines(decoded):
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def clean_generated_text(text: str) -> str:
    text = text.strip()
    for stop in STOP_STRINGS:
        text = text.replace(stop, "")
    text = re.sub(r"^```.*?\n", "", text, flags=re.DOTALL).replace("```", "").strip()
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(marker in line.lower() for marker in MARKER_PATTERNS):
            continue
        line = re.sub(r"^[-*\d.)\s]+", "", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def extract_experience_keywords(experience: str, limit: int = 10) -> List[str]:
    text = re.sub(r"[.,!?;:()\[\]{}\"'“”‘’]", " ", experience)
    words = [w.strip() for w in text.split() if len(w.strip()) >= 2]
    suffixes = ("에서", "에게", "으로", "처럼", "까지", "부터", "하고", "이며", "이랑", "랑", "은", "는", "이", "가", "을", "를", "의", "에", "도")
    result: List[str] = []
    for word in words:
        candidates = [word]
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                candidates.append(word[: -len(suffix)])
        for candidate in candidates:
            if candidate and candidate not in result:
                result.append(candidate)
    return result[:limit]


def looks_like_broken_korean(poem: str) -> bool:
    compact = poem.replace(" ", "").replace("\n", "")
    if not compact:
        return True
    hangul_chars = len(re.findall(r"[가-힣]", compact))
    if len(compact) >= 20 and hangul_chars / max(len(compact), 1) < 0.45:
        return True
    if re.search(r"[A-Za-z]{12,}", poem):
        return True
    if re.search(r"(.)\1{8,}", compact):
        return True
    return False


def validate_poem(poem: str, experience: str) -> Tuple[str, str]:
    if not poem.strip():
        return "invalid_shown", "empty_output"
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    if len(lines) != 3:
        return "invalid_shown", "not_three_lines"
    if any(sum(line.count(pattern) for pattern in PROSE_PATTERNS) > 0 for line in lines):
        return "invalid_shown", "prose_report_like_output"
    if any(len(line) > 70 for line in lines):
        return "invalid_shown", "prose_report_like_output"
    if looks_like_broken_korean(poem):
        return "invalid_shown", "broken_korean_like_output"
    keywords = extract_experience_keywords(experience, limit=8)
    if keywords:
        compact_poem = poem.replace(" ", "")
        matched = [kw for kw in keywords if kw.replace(" ", "") in compact_poem]
        if not matched:
            return "invalid_shown", "no_experience_keyword_match"
    return "valid", "ok"


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app_version": APP_VERSION,
        "model": MODEL_ID,
        "base_model": BASE_MODEL_ID,
        "mode": "finetuned_experiment",
        "show_invalid_outputs": SHOW_INVALID_OUTPUTS,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    started = time.time()
    request_id = str(uuid.uuid4())
    params = build_generation_params(req.languageJump)

    raw_output = generate_with_custom_decoding(req.experience, params)
    poem = clean_generated_text(raw_output)
    validation_status, validation_reason = validate_poem(poem, req.experience)

    elapsed = round(time.time() - started, 3)
    params.update(
        {
            "request_id": request_id,
            "elapsed_seconds": elapsed,
            "strategy": "custom_logits_sampling",
            "validation_status": validation_status,
            "validation_reason": validation_reason,
        }
    )

    print(
        f"[generate][{request_id}] status={validation_status} reason={validation_reason} elapsed={elapsed}s",
        flush=True,
    )

    return GenerateResponse(
        poem=poem,
        mode="finetuned_experiment",
        model=MODEL_ID,
        validation_status=validation_status,
        validation_reason=validation_reason,
        params=params,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
