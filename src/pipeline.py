"""Orquestador del pipeline de datos (ticket 2.3).

Ejecuta de extremo a extremo y en orden los módulos del pipeline:

1. ``ingesta``            -> ``src.data.load_data``        (ticket 1.2)
2. ``hardware_motor``     -> ``src.features.build_hw``     (ticket 2.1)
3. ``target_resiliencia`` -> ``src.features.build_target`` (ticket 2.2)

Cada paso se ejecuta como ``python -m <módulo>`` desde la raíz del repo, así que
las rutas no dependen del directorio desde el que se lance el comando.

Los pasos cuyo script todavía no existe (porque otro ticket no lo ha entregado)
se reportan como PENDIENTE y se omiten sin romper el flujo. Con ``--strict`` se
tratan como error (útil en CI o en el despliegue, cuando ya debe estar todo).

Uso::

    uv run python -m src.pipeline                 # pipeline completo
    uv run python -m src.pipeline --list          # pasos y su estado
    uv run python -m src.pipeline --dry-run       # muestra qué correría
    uv run python -m src.pipeline --only ingesta  # un solo paso
    uv run python -m src.pipeline --strict        # los pendientes fallan

Código de salida: 0 = bien, 1 = falló algún paso, 2 = hay pasos pendientes y se
usó ``--strict``.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = ("data/interim", "data/processed", "data/features")


@dataclass(frozen=True)
class Step:
    """Un paso del pipeline, ejecutado como ``python -m <module>``."""

    name: str
    module: str
    script: str  # ruta relativa a la raíz; si no existe, el paso está pendiente
    description: str
    args: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()  # archivos que deben existir antes de correr
    outputs: tuple[str, ...] = ()  # archivos que deben existir después
    hint: str = ""  # ayuda si faltan las entradas


class Status(str, Enum):
    OK = "OK"
    PENDING = "PENDIENTE"
    FAILED = "FALLÓ"
    SKIPPED = "OMITIDO"
    DRY_RUN = "SIMULADO"


@dataclass
class StepResult:
    step: Step
    status: Status
    seconds: float = 0.0
    detail: str = ""


# Completar `outputs` de hardware_motor cuando el ticket 2.1 fije sus nombres.
STEPS: tuple[Step, ...] = (
    Step(
        name="ingesta",
        module="src.data.load_data",
        script="src/data/load_data.py",
        description="Excel crudo -> cohortes en data/interim/ (ticket 1.2)",
        inputs=("data/raw/kmc400_datos_completos.xlsx",),
        outputs=(
            "data/interim/canguro.parquet",
            "data/interim/tradicional.parquet",
            "data/interim/referencia.parquet",
            "data/interim/referencia_rciu_fenton.parquet",
        ),
        hint="Descarga los datos primero: uv run dvc pull",
    ),
    Step(
        name="hardware_motor",
        module="src.features.build_hw",
        script="src/features/build_hw.py",
        description="Features de hardware motor, ROIs e ICV (ticket 2.1)",
    ),
    Step(
        name="target_resiliencia",
        module="src.features.build_target",
        script="src/features/build_target.py",
        description="SW_motor y target de resiliencia (ticket 2.2)",
        outputs=("data/features/target_resilience.parquet",),
    ),
)


def is_available(step: Step, root: Path = ROOT) -> bool:
    """El paso está disponible si su script ya existe en el repo."""
    return (root / step.script).is_file()


def select_steps(steps: Sequence[Step], only: Sequence[str] | None) -> list[Step]:
    """Filtra por nombre conservando el orden del pipeline."""
    if not only:
        return list(steps)
    known = {s.name for s in steps}
    unknown = [n for n in only if n not in known]
    if unknown:
        raise ValueError(
            f"Paso(s) desconocido(s): {unknown}. Disponibles: {sorted(known)}"
        )
    return [s for s in steps if s.name in only]


def ensure_dirs(root: Path = ROOT) -> None:
    """Crea las carpetas de datos si no existen (ejecución desde cero)."""
    for rel in DATA_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)


def run_step(step: Step, root: Path = ROOT, *, dry_run: bool = False) -> StepResult:
    """Ejecuta un paso y verifica sus entradas y salidas."""
    if not is_available(step, root):
        return StepResult(step, Status.PENDING, detail=f"falta {step.script}")

    missing_in = [p for p in step.inputs if not (root / p).exists()]
    if missing_in:
        hint = f" ({step.hint})" if step.hint else ""
        return StepResult(
            step, Status.FAILED, detail=f"faltan entradas: {missing_in}{hint}"
        )

    cmd = [sys.executable, "-m", step.module, *step.args]
    if dry_run:
        return StepResult(step, Status.DRY_RUN, detail=" ".join(cmd[1:]))

    logger.info("Paso '%s': %s", step.name, " ".join(cmd[1:]))
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=root, check=False)
    seconds = time.perf_counter() - start

    if proc.returncode != 0:
        return StepResult(
            step, Status.FAILED, seconds, f"código de salida {proc.returncode}"
        )
    missing_out = [p for p in step.outputs if not (root / p).exists()]
    if missing_out:
        return StepResult(
            step, Status.FAILED, seconds, f"no generó las salidas: {missing_out}"
        )
    return StepResult(step, Status.OK, seconds)


def run_pipeline(
    steps: Sequence[Step],
    root: Path = ROOT,
    *,
    dry_run: bool = False,
    strict: bool = False,
) -> tuple[int, list[StepResult]]:
    """Corre los pasos en orden. Se detiene en el primer fallo."""
    ensure_dirs(root)
    results: list[StepResult] = []
    failed = False
    for step in steps:
        if failed:
            results.append(
                StepResult(step, Status.SKIPPED, detail="falló un paso previo")
            )
            continue
        result = run_step(step, root, dry_run=dry_run)
        results.append(result)
        if result.status is Status.FAILED:
            failed = True

    if failed:
        code = 1
    elif strict and any(r.status is Status.PENDING for r in results):
        code = 2
    else:
        code = 0
    return code, results


def format_summary(results: Sequence[StepResult]) -> str:
    """Tabla de resultados para imprimir al final."""
    lines = [f"{'paso':22s}{'estado':11s}{'seg':>6s}  detalle"]
    for r in results:
        lines.append(
            f"{r.step.name:22s}{r.status.value:11s}{r.seconds:6.1f}  {r.detail}"
        )
    return "\n".join(lines)


def format_list(steps: Sequence[Step], root: Path = ROOT) -> str:
    """Pasos del pipeline y si su script ya existe."""
    lines = [f"{'paso':22s}{'estado':13s}descripción"]
    for s in steps:
        estado = "disponible" if is_available(s, root) else "pendiente"
        lines.append(f"{s.name:22s}{estado:13s}{s.description}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline de datos de extremo a extremo")
    p.add_argument("--list", action="store_true", help="lista los pasos y su estado")
    p.add_argument(
        "--only", action="append", metavar="PASO", help="corre solo este paso"
    )
    p.add_argument("--dry-run", action="store_true", help="no ejecuta, solo muestra")
    p.add_argument("--strict", action="store_true", help="los pasos pendientes fallan")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if args.list:
        print(format_list(STEPS))
        return 0

    try:
        steps = select_steps(STEPS, args.only)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    code, results = run_pipeline(steps, dry_run=args.dry_run, strict=args.strict)
    print()
    print(format_summary(results))
    if code == 1:
        logger.error("El pipeline falló.")
    elif code == 2:
        logger.error("Hay pasos pendientes y se usó --strict.")
    elif any(r.status is Status.PENDING for r in results):
        logger.warning("Hay pasos pendientes (otros tickets): se omitieron.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
