"""Pruebas del orquestador del pipeline (ticket 2.3). No usan datos reales."""

from pathlib import Path

import pytest

from src import pipeline
from src.pipeline import Status, Step


def make_pkg(root: Path, name: str, body: str) -> str:
    """Crea src/fake/<name>.py en una raíz temporal y devuelve su ruta relativa."""
    pkg = root / "src" / "fake"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "src" / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    (pkg / f"{name}.py").write_text(body, encoding="utf-8")
    return f"src/fake/{name}.py"


def fake_step(root: Path, name: str, body: str, **kwargs) -> Step:
    script = make_pkg(root, name, body)
    return Step(
        name=name,
        module=f"src.fake.{name}",
        script=script,
        description=f"paso de prueba {name}",
        **kwargs,
    )


WRITE_OUT = "from pathlib import Path\nPath('out.txt').write_text('ok')\n"


def test_default_steps_are_well_formed():
    names = [s.name for s in pipeline.STEPS]
    assert len(names) == len(set(names))
    assert names == ["ingesta", "hardware_motor", "target_resiliencia", "imputacion"]
    # El paso de ingesta ya existe en el repo (ticket 1.2)
    ingesta = pipeline.STEPS[0]
    assert pipeline.is_available(ingesta)
    assert pipeline.ROOT.joinpath("src").is_dir()


def test_pending_step_does_not_fail(tmp_path):
    step = Step("falta", "src.fake.no_existe", "src/fake/no_existe.py", "pendiente")
    code, results = pipeline.run_pipeline([step], tmp_path)
    assert code == 0
    assert results[0].status is Status.PENDING


def test_pending_step_fails_with_strict(tmp_path):
    step = Step("falta", "src.fake.no_existe", "src/fake/no_existe.py", "pendiente")
    code, _ = pipeline.run_pipeline([step], tmp_path, strict=True)
    assert code == 2


def test_runs_step_and_checks_outputs(tmp_path):
    step = fake_step(tmp_path, "ok", WRITE_OUT, outputs=("out.txt",))
    code, results = pipeline.run_pipeline([step], tmp_path)
    assert code == 0
    assert results[0].status is Status.OK
    assert (tmp_path / "out.txt").read_text() == "ok"


def test_creates_data_dirs_from_scratch(tmp_path):
    pipeline.run_pipeline([], tmp_path)
    for rel in pipeline.DATA_DIRS:
        assert (tmp_path / rel).is_dir()


def test_failing_step_stops_the_rest(tmp_path):
    bad = fake_step(tmp_path, "mal", "raise SystemExit(3)\n")
    good = fake_step(tmp_path, "bien", WRITE_OUT, outputs=("out.txt",))
    code, results = pipeline.run_pipeline([bad, good], tmp_path)
    assert code == 1
    assert [r.status for r in results] == [Status.FAILED, Status.SKIPPED]
    assert not (tmp_path / "out.txt").exists()


def test_missing_declared_output_is_a_failure(tmp_path):
    step = fake_step(tmp_path, "sin_salida", "pass\n", outputs=("no_existe.txt",))
    code, results = pipeline.run_pipeline([step], tmp_path)
    assert code == 1
    assert "no generó" in results[0].detail


def test_missing_inputs_fail_with_hint(tmp_path):
    step = fake_step(
        tmp_path, "con_entrada", "pass\n", inputs=("raw.xlsx",), hint="corre dvc pull"
    )
    code, results = pipeline.run_pipeline([step], tmp_path)
    assert code == 1
    assert "dvc pull" in results[0].detail


def test_dry_run_does_not_execute(tmp_path):
    step = fake_step(tmp_path, "seco", WRITE_OUT, outputs=("out.txt",))
    code, results = pipeline.run_pipeline([step], tmp_path, dry_run=True)
    assert code == 0
    assert results[0].status is Status.DRY_RUN
    assert not (tmp_path / "out.txt").exists()


def test_select_steps_keeps_order_and_rejects_unknown():
    chosen = pipeline.select_steps(pipeline.STEPS, ["target_resiliencia", "ingesta"])
    assert [s.name for s in chosen] == ["ingesta", "target_resiliencia"]
    with pytest.raises(ValueError, match="desconocido"):
        pipeline.select_steps(pipeline.STEPS, ["inexistente"])


def test_list_and_summary_render():
    text = pipeline.format_list(pipeline.STEPS)
    assert "ingesta" in text and "hardware_motor" in text
    summary = pipeline.format_summary(
        [pipeline.StepResult(pipeline.STEPS[0], Status.OK, 1.5)]
    )
    assert "OK" in summary
