import json

from emtom.task_gen.seed_selector import SeedSelectionConfig, build_seed_candidates


def _write_task(path, title, calibration=None, category="cooperative", tom_level=2):
    data = {
        "title": title,
        "task": f"Task for {title}",
        "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
        "category": category,
        "tom_level": tom_level,
        "calibration": calibration or [],
    }
    path.write_text(json.dumps(data))


def _standard_calibration(model, passed, progress):
    return [
        {
            "model": model,
            "run_mode": "standard",
            "agent_models": {"agent_0": model, "agent_1": model},
            "results": {
                "passed": passed,
                "progress": progress,
            },
        }
    ]


def test_seed_selector_prefers_harder_tasks_when_pool_is_too_easy(tmp_path):
    model = "gpt-5.2"
    _write_task(tmp_path / "hard.json", "hard", _standard_calibration(model, False, 0.35))
    _write_task(tmp_path / "easy.json", "easy", _standard_calibration(model, True, 1.0))
    _write_task(tmp_path / "unknown.json", "unknown")

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
        target_pass_rate=0.20,
        current_pass_rate=0.50,
    )
    weights = {candidate.path.name: candidate.weight for candidate in build_seed_candidates(config)}

    assert weights["hard.json"] > weights["unknown.json"] > weights["easy.json"]


def test_seed_selector_prefers_easier_tasks_when_pool_is_too_hard(tmp_path):
    model = "gpt-5.2"
    _write_task(tmp_path / "hard.json", "hard", _standard_calibration(model, False, 0.10))
    _write_task(tmp_path / "easy.json", "easy", _standard_calibration(model, True, 1.0))

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
        target_pass_rate=0.20,
        current_pass_rate=0.05,
    )
    weights = {candidate.path.name: candidate.weight for candidate in build_seed_candidates(config)}

    assert weights["easy.json"] > weights["hard.json"]


def test_seed_selector_applies_category_and_tom_filters_as_soft_biases(tmp_path):
    model = "gpt-5.2"
    calibration = _standard_calibration(model, False, 0.25)
    _write_task(tmp_path / "match.json", "match", calibration, category="mixed", tom_level=3)
    _write_task(tmp_path / "mismatch.json", "mismatch", calibration, category="cooperative", tom_level=1)

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
        target_pass_rate=0.20,
        current_pass_rate=0.40,
        category="mixed",
        tom_level=3,
    )
    weights = {candidate.path.name: candidate.weight for candidate in build_seed_candidates(config)}

    assert weights["match.json"] > weights["mismatch.json"]
