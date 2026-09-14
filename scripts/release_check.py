"""Generate a test and dependency manifest without usernames, secrets or host paths."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree


def main():
    Path("data").mkdir(exist_ok=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=data/pytest.xml"], check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    lint = subprocess.run([sys.executable, "-m", "ruff", "check", "ai_xr", "experiments", "scripts", "tests"], check=False)
    if lint.returncode:
        raise SystemExit(lint.returncode)
    suites = ElementTree.parse("data/pytest.xml").getroot().findall("testsuite")
    names = ["fastapi", "uvicorn", "httpx", "pydantic-settings", "numpy", "scikit-learn", "pypdf", "python-docx",
             "python-multipart", "Pillow", "rank-bm25", "torch", "transformers", "peft", "accelerate",
             "ultralytics", "sentence-transformers", "mcp", "faster-whisper", "pyttsx3", "pytest", "pytest-asyncio", "ruff"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    report = {"python": platform.python_version(), "os": platform.system(), "cpu": platform.processor() or "not reported",
              "tests": sum(int(s.attrib["tests"]) for s in suites),
              "failures": sum(int(s.attrib["failures"]) for s in suites),
              "errors": sum(int(s.attrib["errors"]) for s in suites), "ruff": "passed", "dependencies": versions}
    Path("evidence/environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: val for key, val in report.items() if key != "dependencies"}))


if __name__ == "__main__":
    main()
