const DEFAULT_HF_BACKEND_URL = "https://z-unghyun-poem-generator-backend.hf.space/generate";

function clampNumber(value, min = 0, max = 100, fallback = 50) {
  const number = Number(value);
  if (Number.isNaN(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function failurePayload(message, payload, extra = {}) {
  return {
    poem: `생성 실패: ${message}`,
    mode: payload.mode,
    model: "Hugging Face backend",
    params: {
      mode: payload.mode === "experiment" ? "custom_decoding" : "prompt_instruction",
      validation_status: "proxy_failure",
      validation_reason: message,
      ...extra
    }
  };
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "POST 요청만 지원합니다." });
  }

  const backendUrl = process.env.HF_BACKEND_URL || DEFAULT_HF_BACKEND_URL;
  const hfToken = process.env.HF_BACKEND_TOKEN;

  const payload = {
    mode: req.body?.mode === "experiment" ? "experiment" : "prompt",
    experience: String(req.body?.experience || "").slice(0, 2000),
    experienceDensity: clampNumber(req.body?.experienceDensity, 0, 100, 80),
    languageJump: clampNumber(req.body?.languageJump, 0, 100, 65),
    dadaIntensity: clampNumber(req.body?.dadaIntensity, 0, 100, 40)
  };

  try {
    const headers = { "Content-Type": "application/json" };
    if (hfToken) headers.Authorization = `Bearer ${hfToken}`;

    const response = await fetch(backendUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(payload)
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      return res.status(200).json(
        failurePayload(
          data?.detail || data?.error || "Hugging Face backend 요청에 실패했습니다.",
          payload,
          { backend_status: response.status, backend_url: backendUrl }
        )
      );
    }

    return res.status(200).json({
      poem: data?.poem || data?.text || "생성 실패: 빈 응답이 반환되었습니다.",
      mode: data?.mode || payload.mode,
      model: data?.model || "Hugging Face backend",
      params: data?.params || {
        validation_status: "missing_params",
        validation_reason: "backend response did not include params"
      }
    });
  } catch (error) {
    return res.status(200).json(
      failurePayload(error?.message || "Hugging Face backend 연결 중 오류가 발생했습니다.", payload, { backend_url: backendUrl })
    );
  }
}
