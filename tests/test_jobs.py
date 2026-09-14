from ai_xr.jobs import extract_skills, llm_skill_evidence, rank_jobs


def test_skill_token_boundaries_and_aliases():
    result = extract_skills("I use C#, C++, FastAPI, Python and 检索增强, not a random variable.")
    assert {"C#", "C++", "FastAPI", "Python", "RAG"} <= result.keys()
    assert "XR" not in result and "Git" not in result


def test_weighted_coverage_and_gap():
    jobs = [{"id": "j", "title": "AI", "description": "Python FastAPI Docker", "required": ["Python", "FastAPI"], "preferred": ["Docker"]}]
    result = rank_jobs("Python and Docker", jobs)[0]
    assert result["score"] == 60 and result["missing_required"] == ["FastAPI"]
    assert all(e in "Python and Docker" for e in result["evidence"].values())


def test_unknown_requirements_are_not_perfect_score():
    result = rank_jobs("Python", [{"id":"j","title":"Job","description":"Work with people"}])[0]
    assert result["score"] is None


async def test_llm_extraction_requires_exact_source_evidence():
    class Provider:
        mode = "test"
        async def chat(self, messages):
            return {"content": '{"skills":[{"name":"CUDA","evidence":"Python"},{"name":"Python","evidence":"Python"},{"name":"RAG","evidence":"invented"}]}'}
    assert (await llm_skill_evidence("Python", Provider()))["skills"] == {"Python": "Python"}
