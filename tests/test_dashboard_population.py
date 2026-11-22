"""Pruebas de la pestaña poblacional del dashboard (ticket 5.3).

Usan los reportes agregados versionados en ``reports/``, así que corren en CI.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import population as pop

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reports() -> pop.PopulationReports:
    return pop.load_reports(ROOT / "reports")


def test_missing_reports_raise_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Faltan reportes"):
        pop.load_reports(tmp_path)


def test_reports_have_no_subject_level_columns(reports: pop.PopulationReports) -> None:
    frames = [
        reports.tabla1,
        reports.nhpt,
        reports.contraste,
        reports.asociacion,
        reports.etapas,
        reports.incrementos,
    ]
    for frame in frames:
        assert not {"code", "BirthDate"} & set(frame.columns)


def test_group_sizes_follow_dec040(reports: pop.PopulationReports) -> None:
    sizes = pop.group_sizes(reports.tabla1)
    assert sizes == {
        "referencia": 28,
        "referencia_completa": 37,
        "prematuros": 209,
        "prematuros_qc": 168,
    }


def test_main_label_counts(reports: pop.PopulationReports) -> None:
    assert pop.main_label_counts(reports.calibracion) == (98, 209)


def test_cohort_table_has_one_row_per_variable(reports: pop.PopulationReports) -> None:
    table = pop.cohort_table(reports.tabla1)
    assert list(table["Variable"]) == [
        pop.TABLE1_LABELS[v]
        for v in reports.tabla1["variable"]
        if v in pop.TABLE1_LABELS
    ]
    assert table.notna().all().all()


def test_nhpt_by_group_uses_reference_without_rciu(
    reports: pop.PopulationReports,
) -> None:
    table = pop.nhpt_by_group(reports.nhpt)
    assert list(table["Grupo"]) == list(pop.NHPT_GROUPS.values())
    reference = table.loc[table["Grupo"] == "Término (sin RCIU)"].iloc[0]
    assert reference["n"] == 37
    assert reference["hedges_g_vs_ref"] == pytest.approx(0.0)


def test_structure_forest_has_the_12_main_volumes(
    reports: pop.PopulationReports,
) -> None:
    forest = pop.structure_forest(reports.contraste)
    assert len(forest) == 12
    assert forest["hedges_g"].is_monotonic_increasing
    assert (forest["hedges_g_ci95_low"] <= forest["hedges_g_ci95_high"]).all()


def test_structure_nhpt_shows_the_12_main_volumes(reports: pop.PopulationReports) -> None:
    table = pop.structure_nhpt_table(reports.asociacion, reports.contraste)
    assert len(table) == 12
    assert (table["IC inf."] <= table["r parcial"]).all()
    assert (table["r parcial"] <= table["IC sup."]).all()


def test_roi_labels() -> None:
    assert pop.roi_label("sub_lh_caudate_vol") == "Caudado izq. (volumen)"
    assert pop.roi_label("cst_rh_fa") == "Corticoespinal der. (FA)"
    assert pop.roi_label("ba_lh_ba4a_area") == "BA4a (M1) izq. (área)"
    assert pop.roi_label("icv") == "Volumen intracraneal"


def test_calibration_table_labels_every_criterion(
    reports: pop.PopulationReports,
) -> None:
    table = pop.calibration_table(reports.calibracion)
    assert len(table) == len(reports.calibracion["calibration_candidates"])
    assert table["Criterio"].iloc[0].startswith("Principal")


@pytest.mark.parametrize(
    ("target", "scenario"),
    [
        ("hardware_ok", "principal"),
        ("nhpt", "principal"),
        ("hardware_ok_sex", "etiqueta_sexo"),
        ("hardware_ok_ref_completa", "etiqueta_ref_completa"),
        ("hardware_ok", "ipw"),
        ("nhpt_mri", "submuestra_rm"),
    ],
)
def test_stage_curve_orders_stages(
    reports: pop.PopulationReports, target: str, scenario: str
) -> None:
    curve = pop.stage_curve(reports.etapas, target, scenario)
    assert len(curve) == 12  # 6 etapas x 2 modelos
    for _, model in curve.groupby("Modelo"):
        assert list(model["orden"]) == list(range(6))


def test_stage_curve_unknown_scenario_raises(reports: pop.PopulationReports) -> None:
    with pytest.raises(ValueError, match="No hay resultados"):
        pop.stage_curve(reports.etapas, "hardware_ok", "no_existe")


def test_population_page_renders_without_errors() -> None:
    app = AppTest.from_file(
        str(ROOT / "dashboard" / "poblacional.py"), default_timeout=60
    )
    app.run()
    assert not app.exception
    assert app.title[0].value == "Análisis poblacional"
    assert len(app.metric) == 4


def test_population_page_switches_to_nhpt() -> None:
    app = AppTest.from_file(
        str(ROOT / "dashboard" / "poblacional.py"), default_timeout=60
    )
    app.run()
    app.radio[0].set_value("nhpt").run()
    assert not app.exception
    assert app.selectbox[0].options == ["Principal", "Submuestra con resonancia"]


def test_entrypoint_defaults_to_population_page() -> None:
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Análisis poblacional"


def test_curve_values_are_aggregates(reports: pop.PopulationReports) -> None:
    curve = pop.stage_curve(reports.etapas, "hardware_ok")
    assert pd.api.types.is_float_dtype(curve["media"])


def test_icv_secondary_analysis_is_loaded(reports: pop.PopulationReports) -> None:
    assert reports.icv is not None and reports.icv_meta is not None
    curve = pop.icv_curve(reports.icv)
    assert len(curve) == 18  # 6 etapas x 3 modelos
    assert set(curve["Modelo"]) == set(pop.MODEL_LABELS.values())
    readings = pop.icv_reading(reports.icv_meta)
    assert list(readings) == list(pop.STAGE_LABELS.values())
    assert set(readings.values()) <= set(pop.ICV_READINGS)


def test_icv_curve_unknown_label_raises(reports: pop.PopulationReports) -> None:
    assert reports.icv is not None
    with pytest.raises(ValueError, match="5.6"):
        pop.icv_curve(reports.icv, "no_existe")


def test_icv_is_optional(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    shutil.rmtree(tmp_path / "reports" / "incremental" / "icv")
    loaded = pop.load_reports(tmp_path / "reports")
    assert loaded.icv is None and loaded.icv_meta is None


def test_calibration_includes_icv_labels(reports: pop.PopulationReports) -> None:
    table = pop.calibration_table(reports.calibracion)
    assert "Ajustada por ICV y sexo (5.6, secundario)" in set(table["Criterio"])


def test_population_page_shows_icv_scenario() -> None:
    app = AppTest.from_file(
        str(ROOT / "dashboard" / "poblacional.py"), default_timeout=60
    )
    app.run()
    app.selectbox[0].set_value("icv").run()
    assert not app.exception
    assert any("Lectura prerregistrada" in m.value for m in app.markdown)


def test_page_refreshes_reports_and_hides_absent_sensitivities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    directory = tmp_path / "reports"
    shutil.copytree(ROOT / "reports", directory, ignore=shutil.ignore_patterns("icv"))
    monkeypatch.setenv("DASHBOARD_REPORTS_DIR", str(directory))
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    assert app.metric[0].value == "28"
    assert "Ponderado por IPW" in app.selectbox[0].options

    cohorts_path = directory / pop.HARDWARE_FILES["tabla1"]
    cohorts = pd.read_csv(cohorts_path)
    cohorts.loc[cohorts["variable"].eq("BIRTH_peso5"), "n_ref_mri"] = 29
    cohorts.to_csv(cohorts_path, index=False)
    stages_path = directory / pop.INCREMENTAL_FILES["etapas"]
    stages = pd.read_csv(stages_path)
    stages.loc[stages["escenario"].eq("principal")].to_csv(stages_path, index=False)

    app.run()
    assert not app.exception
    assert app.metric[0].value == "29"
    assert app.selectbox[0].options == ["Principal"]
