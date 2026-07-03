const DEFAULT_HF_BACKEND_URL = "https://z-unghyun-poem-generator-backend.hf.space/generate";
const FIXED_MODE = "finetuned_experiment";
const MAX_EXPERIENCE_CHARS = 2000;

function clampNumber(value, min = 0, max = 100, fallback = 65) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, Math.round(number)));
}

function normalizeBackendUrl(url) {
  const raw = String(url || DEFAULT_HF_BACKEND_URL).trim();
  const withoutTrailingSlash = raw.replace(/\/+$/, "");
  if (withoutTrailingSlash.endsWith("/generate")) {
    return withoutTrailingSlash;
  }
  return `${withoutTrailingSlash}/generate`;
}

function getRequestBody(req) {
  if (!req.body) return {};

  if (typeof req.body === "string") {
    try {
      return JSON.parse(req.body);
    } catch {
      return {};
    }
  }

  return req.body;
}

async function readBackendResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const rawText = await response.text();

  if (contentType.includes("application/json")) {
    try {
      return {
        data: JSON.parse(rawText),
        rawText
      };
    } catch {
      return {
        data: null,
        rawText
      };
    }
  }

  return {
    data: null,
    rawText
  };
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({
      error: "method_not_allowed",
      message: "POST 요청만 지원합니다."
    });
  }

  const body = getRequestBody(req);
  const backendUrl = normalizeBackendUrl(process.env.HF_BACKEND_URL || DEFAULT_HF_BACKEND_URL);
  const hfToken = process.env.HF_BACKEND_TOKEN;

  const payload = {
    experience: String(body.experience || "").slice(0, MAX_EXPERIENCE_CHARS),
    languageJump: clampNumber(body.languageJump, 0, 100, 65)
  };

  if (!payload.experience.trim()) {
    return res.status(400).json({
      error: "empty_experience",
      message: "experience가 비어 있습니다.",
      mode: FIXED_MODE,
      params: {
        validation_status: "proxy_rejected",
        validation_reason: "empty_experience"
      }
    });
  }

  const headers = {
    "Content-Type": "application/json"
  };

  if (hfToken) {
    headers.Authorization = `Bearer ${hfToken}`;
  }

  try {
    const response = await fetch(backendUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(payload)
    });

    const { data, rawText } = await readBackendResponse(response);

    if (!response.ok) {
      return res.status(response.status).json({
        error: "backend_error",
        backend_status: response.status,
        backend_url: backendUrl,
        backend_response: data ?? rawText ?? null,
        forwarded_payload: payload
      });
    }

    if (data && typeof data === "object") {
      return res.status(200).json({
        ...data,
        mode: data.mode || FIXED_MODE
      });
    }

    return res.status(502).json({
      error: "invalid_backend_response",
      message: "Hugging Face backend가 JSON 응답을 반환하지 않았습니다.",
      backend_status: response.status,
      backend_url: backendUrl,
      backend_response: rawText ?? null,
      forwarded_payload: payload,
      mode: FIXED_MODE,
      params: {
        validation_status: "proxy_failure",
        validation_reason: "invalid_backend_response"
      }
    });
  } catch (error) {
    return res.status(502).json({
      error: "backend_connection_failed",
      message: error?.message || "Hugging Face backend 연결 중 오류가 발생했습니다.",
      backend_url: backendUrl,
      forwarded_payload: payload,
      mode: FIXED_MODE,
      params: {
        validation_status: "proxy_failure",
        validation_reason: "backend_connection_failed"
      }
    });
  }
}
