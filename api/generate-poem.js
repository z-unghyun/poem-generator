const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

const DEFAULT_MODEL = "qwen/qwen-2.5-7b-instruct";

function clampNumber(value, min = 0, max = 100, fallback = 50) {
  const number = Number(value);
  if (Number.isNaN(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function extractKeywords(text = "") {
  const cleaned = text
    .replace(/[.,!?;:()\[\]{}"'“”‘’]/g, " ")
    .split(/\s+/)
    .map((word) => word.trim())
    .filter((word) => word.length > 1);

  return [...new Set(cleaned)].slice(0, 12);
}

function getSliderTone(value, low, mid, high) {
  if (value < 35) return low;
  if (value < 70) return mid;
  return high;
}

function buildPromptModeMessages(payload) {
  const { experience, experienceDensity, languageJump, dadaIntensity } = payload;

  const experienceRule = getSliderTone(
    experienceDensity,
    "사용자의 경험은 은은한 배경으로만 반영한다.",
    "사용자의 경험 속 장소, 감정, 사물을 여러 행에 반영한다.",
    "사용자의 경험 속 핵심 이미지, 감각, 정서를 시 전체의 중심축으로 삼는다."
  );

  const jumpRule = getSliderTone(
    languageJump,
    "문장은 비교적 자연스럽고 서정적으로 유지한다.",
    "평범한 표현을 피하고 낯선 은유와 감각적 비약을 사용한다.",
    "관습적 연결을 적극적으로 피하고, 의미가 닿을 듯 말 듯한 강한 언어적 도약을 만든다."
  );

  const dadaRule = getSliderTone(
    dadaIntensity,
    "문법과 의미는 대부분 보존한다.",
    "일부 반복, 절단, 행갈이를 활용하되 전체 의미는 유지한다.",
    "반복, 절단, 비문, 우연적 조합을 적극 허용한다. 단, 완전한 무의미로 붕괴시키지는 않는다."
  );

  return [
    {
      role: "system",
      content: "너는 한국어 현대시와 실험시를 생성하는 AI 시 생성기다. 출력은 오직 시 본문만 작성한다. 해설, 제목, 따옴표, 목록, 설명문은 쓰지 않는다."
    },
    {
      role: "user",
      content: `다음 삶의 경험을 바탕으로 6~10행의 한국어 시를 써라.\n\n[삶의 경험]\n${experience || "구체적인 경험이 입력되지 않았다. 비어 있음 자체를 경험으로 삼아 시를 쓴다."}\n\n[슬라이더]\n- 경험 밀도: ${experienceDensity}/100\n- 언어 도약도: ${languageJump}/100\n- 다다 강도: ${dadaIntensity}/100\n\n[생성 규칙]\n1. ${experienceRule}\n2. ${jumpRule}\n3. ${dadaRule}\n4. 추상적인 설명보다 이미지, 감각, 사물, 움직임을 중심으로 쓴다.\n5. 출력은 시 본문만 작성한다.`
    }
  ];
}

function buildExperimentModeMessages(payload) {
  const { experience, experienceDensity, languageJump, dadaIntensity } = payload;
  const keywords = extractKeywords(experience);
  const keywordText = keywords.length ? keywords.join(", ") : "비어 있음, 무명, 빈칸, 침묵";

  return [
    {
      role: "system",
      content: "너는 확률분포를 실험적으로 왜곡하는 한국어 시 생성기다. 가장 평범하고 예측 가능한 표현을 피하되, 완전히 이해 불가능한 잡음은 만들지 않는다. 출력은 오직 시 본문만 작성한다."
    },
    {
      role: "user",
      content: `사용자의 삶의 경험에서 추출한 핵심어를 의미적 중력장처럼 사용해 6~10행의 한국어 실험시를 써라.\n\n[삶의 경험]\n${experience || "구체적인 경험이 입력되지 않았다. 비어 있음 자체를 경험으로 삼아 시를 쓴다."}\n\n[경험어 후보]\n${keywordText}\n\n[실험 조건]\n- 경험 밀도: ${experienceDensity}/100\n- 언어 도약도: ${languageJump}/100\n- 다다 강도: ${dadaIntensity}/100\n\n[확률 실험 규칙]\n1. 경험 밀도가 높을수록 경험어 후보를 직접 또는 변형된 이미지로 더 자주 호출한다.\n2. 언어 도약도가 높을수록 가장 자연스러운 다음 표현을 피하고 중간 확률대의 낯선 연결을 선택한 것처럼 쓴다.\n3. 다다 강도가 높을수록 반복, 단절, 병치, 비문, 음성적 유사성을 사용한다.\n4. 시는 파열되어도 좋지만, 사용자의 경험에서 출발했다는 흔적은 남긴다.\n5. 출력은 시 본문만 작성한다.`
    }
  ];
}

function getGenerationParams(mode, languageJump, dadaIntensity) {
  if (mode === "experiment") {
    return {
      temperature: Number((0.75 + languageJump * 0.009).toFixed(2)),
      top_p: Number((0.78 + languageJump * 0.002).toFixed(2)),
      frequency_penalty: Number((0.1 + dadaIntensity * 0.006).toFixed(2)),
      presence_penalty: Number((0.05 + languageJump * 0.004).toFixed(2))
    };
  }

  return {
    temperature: Number((0.55 + languageJump * 0.005).toFixed(2)),
    top_p: 0.9,
    frequency_penalty: Number((0.05 + dadaIntensity * 0.002).toFixed(2)),
    presence_penalty: 0.1
  };
}

function applyDadaPostprocess(poem, dadaIntensity) {
  if (!poem || dadaIntensity < 60) return poem;

  const lines = poem
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 3) return poem;

  if (dadaIntensity >= 85) {
    const fractured = [];
    lines.forEach((line, index) => {
      fractured.push(line);
      if (index % 2 === 0) {
        const words = line.split(/\s+/).filter(Boolean);
        if (words.length >= 2) fractured.push(`${words[0]} / ${words[words.length - 1]}`);
      }
    });
    return fractured.slice(0, 12).join("\n");
  }

  const repeated = [...lines];
  repeated.splice(2, 0, lines[1]);
  return repeated.slice(0, 11).join("\n");
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "POST 요청만 지원합니다." });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL || DEFAULT_MODEL;

  if (!apiKey) {
    return res.status(500).json({
      error: "OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다. Vercel Project Settings에서 Environment Variables를 설정하세요."
    });
  }

  const mode = req.body?.mode === "experiment" ? "experiment" : "prompt";
  const payload = {
    mode,
    experience: String(req.body?.experience || "").slice(0, 2000),
    experienceDensity: clampNumber(req.body?.experienceDensity, 0, 100, 80),
    languageJump: clampNumber(req.body?.languageJump, 0, 100, 65),
    dadaIntensity: clampNumber(req.body?.dadaIntensity, 0, 100, 40)
  };

  const messages = mode === "experiment"
    ? buildExperimentModeMessages(payload)
    : buildPromptModeMessages(payload);

  const generationParams = getGenerationParams(mode, payload.languageJump, payload.dadaIntensity);

  try {
    const response = await fetch(OPENROUTER_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://poem-generator.vercel.app",
        "X-Title": "Poem Generator"
      },
      body: JSON.stringify({
        model,
        messages,
        max_tokens: 700,
        ...generationParams
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: data?.error?.message || "OpenRouter API 요청에 실패했습니다."
      });
    }

    const rawPoem = data?.choices?.[0]?.message?.content?.trim() || "";
    const poem = mode === "experiment"
      ? applyDadaPostprocess(rawPoem, payload.dadaIntensity)
      : rawPoem;

    return res.status(200).json({
      poem,
      mode,
      model,
      params: generationParams
    });
  } catch (error) {
    return res.status(500).json({
      error: error?.message || "시 생성 중 알 수 없는 오류가 발생했습니다."
    });
  }
}
