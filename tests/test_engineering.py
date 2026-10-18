import ast
from pathlib import Path
import pytest
from scripts.run_async_evaluation import policy_for, validate_export
from src.schedulers import OraclePolicy


def test_oracle_reference_has_explicit_nonoptimal_provenance():
    policy = policy_for("oracle", 5)
    assert isinstance(policy, OraclePolicy)
    assert policy.schedule == (1, 2, 3, 4, 5)
    assert "not optimized" in policy.describe()["provenance"]
    with pytest.raises(ValueError):
        policy_for("oracle", 5, schedule=[6])


def test_public_api_docstrings():
    missing = []
    for path in Path("src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    missing.append(f"{path}:{node.name}")
    assert not missing


def test_reject_incomplete_export(tmp_path):
    import json

    (tmp_path / "summary.json").write_text(json.dumps({"status": "partial"}))
    (tmp_path / "frames.jsonl").write_text("")
    with pytest.raises(ValueError):
        validate_export(tmp_path, 5)
