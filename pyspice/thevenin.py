"""Thevenin theorem validation with PySpice.

The original resistor network and its Thevenin equivalent are simulated under
open-circuit, short-circuit and loaded conditions.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "pyspice-matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import u_kOhm, u_Ohm, u_V


SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

V1_V = 12.0
R1_OHM = 2_000.0
R2_OHM = 3_000.0
RL_OHM = 1_000.0
R_SHORT_OHM = 1e-6

VTH_THEORY_V = V1_V * R2_OHM / (R1_OHM + R2_OHM)
RTH_THEORY_OHM = R1_OHM * R2_OHM / (R1_OHM + R2_OHM)
ISC_THEORY_A = VTH_THEORY_V / RTH_THEORY_OHM
VL_THEORY_V = VTH_THEORY_V * RL_OHM / (RTH_THEORY_OHM + RL_OHM)
IL_THEORY_A = VL_THEORY_V / RL_OHM


@contextmanager
def ngspice_working_directory():
    """Run Ngspice from its package directory so spinit resolves correctly."""
    original_directory = Path.cwd()
    os.chdir(NgSpiceShared.NGSPICE_PATH)
    try:
        yield
    finally:
        os.chdir(original_directory)


def _node_voltage(analysis, node_name: str) -> float:
    return float(np.asarray(analysis[node_name]).reshape(-1)[0])


def build_circuit(network: str, condition: str) -> Circuit:
    """Build the original or equivalent network for a specified load condition."""
    if network not in {"original", "equivalent"}:
        raise ValueError(f"Unsupported network: {network}")
    if condition not in {"open", "short", "load"}:
        raise ValueError(f"Unsupported condition: {condition}")

    circuit = Circuit(f"Thevenin {network} - {condition}")
    if network == "original":
        circuit.V("1", "vcc", circuit.gnd, V1_V @ u_V)
        circuit.R("1", "vcc", "terminal", (R1_OHM / 1_000.0) @ u_kOhm)
        circuit.R("2", "terminal", circuit.gnd, (R2_OHM / 1_000.0) @ u_kOhm)
    else:
        circuit.V("th", "vth", circuit.gnd, VTH_THEORY_V @ u_V)
        circuit.R("th", "vth", "terminal", (RTH_THEORY_OHM / 1_000.0) @ u_kOhm)

    if condition == "short":
        circuit.R("short", "terminal", circuit.gnd, R_SHORT_OHM @ u_Ohm)
    elif condition == "load":
        circuit.R("load", "terminal", circuit.gnd, (RL_OHM / 1_000.0) @ u_kOhm)
    return circuit


def _simulate_condition(network: str, condition: str) -> float:
    circuit = build_circuit(network, condition)
    analysis = circuit.simulator(temperature=25, nominal_temperature=25).operating_point()
    return _node_voltage(analysis, "terminal")


def run_simulation() -> dict:
    """Run all original/equivalent test conditions."""
    with ngspice_working_directory():
        open_original = _simulate_condition("original", "open")
        open_equivalent = _simulate_condition("equivalent", "open")
        short_original = _simulate_condition("original", "short")
        short_equivalent = _simulate_condition("equivalent", "short")
        load_original = _simulate_condition("original", "load")
        load_equivalent = _simulate_condition("equivalent", "load")

    isc_original = short_original / R_SHORT_OHM
    isc_equivalent = short_equivalent / R_SHORT_OHM
    il_original = load_original / RL_OHM
    il_equivalent = load_equivalent / RL_OHM

    return {
        "open_original_v": open_original,
        "open_equivalent_v": open_equivalent,
        "short_original_v": short_original,
        "short_equivalent_v": short_equivalent,
        "isc_original_a": isc_original,
        "isc_equivalent_a": isc_equivalent,
        "load_original_v": load_original,
        "load_equivalent_v": load_equivalent,
        "il_original_a": il_original,
        "il_equivalent_a": il_equivalent,
    }


def _relative_difference_percent(value_a: float, value_b: float) -> float:
    scale = max(abs(value_a), abs(value_b), 1e-15)
    return abs(value_a - value_b) / scale * 100.0


def _relative_error_percent(simulated: float, theoretical: float) -> float:
    return abs(simulated - theoretical) / max(abs(theoretical), 1e-15) * 100.0


def verify_results(results: dict) -> dict:
    """Verify equivalence and agreement with the analytical solution."""
    differences = {
        "open_circuit_difference_percent": _relative_difference_percent(
            results["open_original_v"], results["open_equivalent_v"]
        ),
        "short_circuit_difference_percent": _relative_difference_percent(
            results["isc_original_a"], results["isc_equivalent_a"]
        ),
        "loaded_voltage_difference_percent": _relative_difference_percent(
            results["load_original_v"], results["load_equivalent_v"]
        ),
        "loaded_current_difference_percent": _relative_difference_percent(
            results["il_original_a"], results["il_equivalent_a"]
        ),
    }
    theory_errors = {
        "vth_theory_error_percent": _relative_error_percent(
            results["open_original_v"], VTH_THEORY_V
        ),
        "isc_theory_error_percent": _relative_error_percent(
            results["isc_original_a"], ISC_THEORY_A
        ),
        "vl_theory_error_percent": _relative_error_percent(
            results["load_original_v"], VL_THEORY_V
        ),
        "il_theory_error_percent": _relative_error_percent(
            results["il_original_a"], IL_THEORY_A
        ),
    }
    checks = {
        **{name: value <= 1.0 for name, value in differences.items()},
        **{name: value <= 1.0 for name, value in theory_errors.items()},
    }
    return {
        "differences": differences,
        "theory_errors": theory_errors,
        "checks": checks,
        "passed": all(checks.values()),
    }


def plot_results(results: dict) -> None:
    """Create a compact comparison figure for report use."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            "Open-Circuit Voltage",
            "V",
            [VTH_THEORY_V, results["open_original_v"], results["open_equivalent_v"]],
        ),
        (
            axes[0, 1],
            "Short-Circuit Current",
            "mA",
            [ISC_THEORY_A * 1e3, results["isc_original_a"] * 1e3, results["isc_equivalent_a"] * 1e3],
        ),
        (
            axes[1, 0],
            "Loaded Voltage",
            "V",
            [VL_THEORY_V, results["load_original_v"], results["load_equivalent_v"]],
        ),
        (
            axes[1, 1],
            "Loaded Current",
            "mA",
            [IL_THEORY_A * 1e3, results["il_original_a"] * 1e3, results["il_equivalent_a"] * 1e3],
        ),
    ]
    labels = ["Theory", "Original", "Equivalent"]
    colors = ["#475569", "#0e7490", "#2563eb"]
    for axis, title, unit, values in panels:
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.22)
        upper = max(values) * 1.22 if max(values) > 0 else 1.0
        axis.set_ylim(0.0, upper)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.025,
                f"{value:.4g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Thevenin Theorem Validation", fontsize=15)
    figure.savefig(ASSETS_DIR / "thevenin_results.png", dpi=180)
    plt.close(figure)


def _source_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 500" role="img" aria-labelledby="title desc">
  <title id="title">Original Thevenin source network</title>
  <desc id="desc">A 12 volt source with a 2 kilohm series resistor, a 3 kilohm shunt resistor and a 1 kilohm load.</desc>
  <rect width="900" height="500" fill="#ffffff"/>
  <g fill="none" stroke="#172554" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="120" cy="245" r="34"/><line x1="108" y1="233" x2="132" y2="257"/><line x1="120" y1="279" x2="120" y2="395"/>
    <line x1="100" y1="395" x2="140" y2="395"/><line x1="108" y1="407" x2="132" y2="407"/><line x1="116" y1="419" x2="124" y2="419"/>
    <line x1="120" y1="211" x2="120" y2="130"/><line x1="120" y1="130" x2="230" y2="130"/>
    <rect x="230" y="108" width="130" height="44" rx="4"/><line x1="360" y1="130" x2="555" y2="130"/>
    <circle cx="555" cy="130" r="6" fill="#0e7490"/><circle cx="555" cy="400" r="6" fill="#0e7490"/>
    <line x1="555" y1="130" x2="555" y2="185"/><rect x="533" y="185" width="44" height="120" rx="4"/>
    <line x1="555" y1="305" x2="555" y2="400"/><line x1="530" y1="400" x2="580" y2="400"/>
    <line x1="548" y1="412" x2="562" y2="412"/><line x1="552" y1="422" x2="558" y2="422"/>
    <line x1="555" y1="130" x2="750" y2="130"/><line x1="750" y1="130" x2="750" y2="185"/>
    <rect x="728" y="185" width="44" height="120" rx="4"/><line x1="750" y1="305" x2="750" y2="400"/>
    <line x1="725" y1="400" x2="775" y2="400"/><line x1="743" y1="412" x2="757" y2="412"/><line x1="747" y1="422" x2="753" y2="422"/>
    <line x1="555" y1="400" x2="750" y2="400"/>
  </g>
  <g fill="#172554" font-family="Arial, Helvetica, sans-serif" font-size="18">
    <text x="65" y="220">V1 = 12 V</text><text x="272" y="92">R1 = 2 kOhm</text><text x="602" y="252">R2 = 3 kOhm</text>
    <text x="790" y="252">RL = 1 kOhm</text><text x="585" y="112">A</text><text x="585" y="416">B</text>
    <text x="450" y="105">Terminals A-B</text>
  </g>
</svg>
"""


def _equivalent_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 500" role="img" aria-labelledby="title desc">
  <title id="title">Thevenin equivalent circuit</title>
  <desc id="desc">A 7.2 volt Thevenin source, a 1.2 kilohm Thevenin resistance and a 1 kilohm load.</desc>
  <rect width="900" height="500" fill="#ffffff"/>
  <g fill="none" stroke="#172554" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="125" cy="245" r="34"/><line x1="112" y1="247" x2="138" y2="247"/><line x1="125" y1="233" x2="125" y2="261"/>
    <line x1="125" y1="279" x2="125" y2="395"/><line x1="105" y1="395" x2="145" y2="395"/><line x1="113" y1="407" x2="137" y2="407"/><line x1="121" y1="419" x2="129" y2="419"/>
    <line x1="125" y1="211" x2="125" y2="130"/><line x1="125" y1="130" x2="245" y2="130"/>
    <rect x="245" y="108" width="150" height="44" rx="4"/><line x1="395" y1="130" x2="610" y2="130"/>
    <circle cx="610" cy="130" r="6" fill="#0e7490"/><circle cx="610" cy="400" r="6" fill="#0e7490"/>
    <line x1="610" y1="130" x2="610" y2="185"/><rect x="588" y="185" width="44" height="120" rx="4"/>
    <line x1="610" y1="305" x2="610" y2="400"/><line x1="585" y1="400" x2="635" y2="400"/><line x1="603" y1="412" x2="617" y2="412"/><line x1="607" y1="422" x2="613" y2="422"/>
    <line x1="610" y1="130" x2="785" y2="130"/><line x1="785" y1="130" x2="785" y2="185"/><rect x="763" y="185" width="44" height="120" rx="4"/>
    <line x1="785" y1="305" x2="785" y2="400"/><line x1="760" y1="400" x2="810" y2="400"/><line x1="778" y1="412" x2="792" y2="412"/><line x1="782" y1="422" x2="788" y2="422"/>
    <line x1="610" y1="400" x2="785" y2="400"/>
  </g>
  <g fill="#172554" font-family="Arial, Helvetica, sans-serif" font-size="18">
    <text x="57" y="220">Vth = 7.2 V</text><text x="274" y="92">Rth = 1.2 kOhm</text><text x="655" y="252">RL = 1 kOhm</text>
    <text x="638" y="112">A</text><text x="638" y="416">B</text><text x="470" y="105">Terminals A-B</text>
  </g>
</svg>
"""


def write_circuit_svgs() -> None:
    (ASSETS_DIR / "thevenin_source.svg").write_text(_source_svg(), encoding="utf-8")
    (ASSETS_DIR / "thevenin_equivalent.svg").write_text(_equivalent_svg(), encoding="utf-8")


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_simulation()
    verification = verify_results(results)
    plot_results(results)
    write_circuit_svgs()

    print("Thevenin Theorem Validation")
    print(f"Theory: Vth={VTH_THEORY_V:.4f} V, Rth={RTH_THEORY_OHM:.1f} Ohm, Isc={ISC_THEORY_A * 1e3:.4f} mA")
    print(f"Theory loaded: VL={VL_THEORY_V:.6f} V, IL={IL_THEORY_A * 1e3:.6f} mA")
    print(
        "Open circuit: "
        f"original={results['open_original_v']:.9f} V, equivalent={results['open_equivalent_v']:.9f} V"
    )
    print(
        "Short circuit: "
        f"original={results['isc_original_a'] * 1e3:.9f} mA, equivalent={results['isc_equivalent_a'] * 1e3:.9f} mA"
    )
    print(
        "Loaded: "
        f"original=(VL={results['load_original_v']:.9f} V, IL={results['il_original_a'] * 1e3:.9f} mA), "
        f"equivalent=(VL={results['load_equivalent_v']:.9f} V, IL={results['il_equivalent_a'] * 1e3:.9f} mA)"
    )
    print("Original vs equivalent differences:")
    for name, value in verification["differences"].items():
        print(f"  {name}: {value:.6f}%")
    print("Original vs analytical errors:")
    for name, value in verification["theory_errors"].items():
        print(f"  {name}: {value:.6f}%")
    for name, passed in verification["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"RESULT: {'PASS' if verification['passed'] else 'FAIL'}")
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
