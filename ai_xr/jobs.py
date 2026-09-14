import hashlib
import json
import re

import httpx
from pydantic import BaseModel, Field, HttpUrl


SKILLS = {
    "Python": ["python"], "C#": ["c#", "c sharp"], "C++": ["c++"], "Unity": ["unity"],
    "LLM": ["llm", "大语言模型", "大模型"], "RAG": ["rag", "检索增强"],
    "Agent": ["agent", "智能体"], "FastAPI": ["fastapi"], "Docker": ["docker"],
    "PyTorch": ["pytorch", "torch"], "YOLO": ["yolo", "yolov8", "yolo11"],
    "OpenCV": ["opencv"], "SLAM": ["slam"], "HoloLens": ["hololens"],
    "SQL": ["sql", "postgresql", "sqlite", "mysql"], "Linux": ["linux"],
    "Git": ["git", "github"], "TTS": ["tts", "语音合成"], "STT": ["stt", "asr", "语音识别"],
    "MCP": ["mcp"], "LoRA": ["lora", "qlora"], "Transformers": ["transformers", "hugging face"],
    "Tool Calling": ["tool calling", "function calling", "工具调用"],
    "React": ["react"], "TypeScript": ["typescript"], "CUDA": ["cuda"],
    "VLM": ["vlm", "视觉语言模型"], "XR": ["xr", "ar", "vr", "mr", "空间计算"],
}


def extract_skills(text):
    evidence = {}
    for canonical, aliases in SKILLS.items():
        for alias in aliases:
            pattern = re.escape(alias)
            if alias[0].isascii():
                pattern = r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])"
            found = re.search(pattern, text, re.I)
            if found:
                evidence[canonical] = text[max(0, found.start()-32):min(len(text), found.end()+48)]
                break
    return evidence


class Job(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    company: str = Field(default="未指定", max_length=200)
    description: str = Field(min_length=1, max_length=20000)
    required: list[str] = Field(default_factory=list, max_length=100)
    preferred: list[str] = Field(default_factory=list, max_length=100)
    source_url: HttpUrl | None = None
    sample: bool = False


def normalize_skill(value):
    if value in SKILLS:
        return value
    matches = list(extract_skills(value))
    return matches[0] if len(matches) == 1 else value.strip()


def rank_jobs(resume, jobs):
    skills = extract_skills(resume)
    results = []
    for raw in jobs:
        job = Job.model_validate(raw)
        required = {normalize_skill(s) for s in job.required}
        preferred = {normalize_skill(s) for s in job.preferred} - required
        inferred = not required and not preferred
        if inferred:
            required = set(extract_skills(job.description))
        matched_required = required.intersection(skills)
        matched_preferred = preferred.intersection(skills)
        denominator = len(required) * 2 + len(preferred)
        score = round(100 * (len(matched_required)*2+len(matched_preferred))/denominator, 1) if denominator else None
        gaps = sorted(required - skills.keys())
        results.append({"job": job.model_dump(mode="json"), "score": score,
                        "matched": sorted(matched_required | matched_preferred), "missing_required": gaps,
                        "missing_preferred": sorted(preferred - skills.keys()),
                        "evidence": {s: skills[s] for s in sorted(matched_required | matched_preferred)},
                        "requirements_inferred": inferred,
                        "suggestions": [f"为 {s} 补充可验证的代码、项目或评测证据；掌握后再加入简历。" for s in gaps],
                        "score_meaning": "技能词条覆盖率，非录用概率；不评估学历、签证或工作年限。"})
    return sorted(results, key=lambda item: item["score"] if item["score"] is not None else -1, reverse=True)


async def llm_skill_evidence(resume, provider):
    if provider.mode == "extractive":
        return {"skills": extract_skills(resume), "mode": "taxonomy-extraction"}
    message = await provider.chat([
        {"role": "system", "content": '从简历中抽取技能。只返回JSON：{"skills":[{"name":"Python","evidence":"原文逐字片段"}]}。不得补全未写明的能力。'},
        {"role": "user", "content": resume},
    ])
    try:
        body = json.loads((message.get("content") or "").strip().removeprefix("```json").removesuffix("```").strip())
        validated = {}
        for item in body["skills"]:
            evidence = item["evidence"]
            if evidence and evidence in resume:
                name = normalize_skill(item["name"])
                if name in extract_skills(evidence):
                    validated[name] = evidence
        if not validated and extract_skills(resume):
            return {"skills": extract_skills(resume), "mode": "taxonomy-fallback",
                    "reason": "model-returned-no-supported-entities"}
        return {"skills": validated, "mode": provider.mode, "validation": "exact-source-substring"}
    except (ValueError, KeyError, TypeError):
        return {"skills": extract_skills(resume), "mode": "taxonomy-fallback", "reason": "invalid-model-json"}


async def search_jobs(query, api_key):
    if not api_key:
        raise ValueError("Configure AI_SEARCH_API_KEY for Brave Search or paste job descriptions locally")
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get("https://api.search.brave.com/res/v1/web/search",
                                    params={"q": query, "count": 8},
                                    headers={"X-Subscription-Token": api_key})
        response.raise_for_status()
        results = []
        for item in response.json().get("web", {}).get("results", []):
            # Search snippets are references, not verified open vacancies.
            results.append({"id": hashlib.sha256(item["url"].encode()).hexdigest()[:16],
                            "title": item["title"], "company": "检索结果待核验",
                            "description": item.get("description") or item["title"],
                            "source_url": item["url"], "sample": False})
        return results
