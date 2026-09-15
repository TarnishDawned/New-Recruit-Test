"""RC low-pass filter simulation and validation using PySpice.

The script generates an editable circuit diagram, transient response and Bode
plots. It is self-contained and can be run from either the repository root or
the pyspice directory.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
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
from PySpice.Unit import u_Hz, u_kOhm, u_ms, u_nF, u_ns, u_us, u_V


SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

R_OHM = 1_000.0
C_FARAD = 100e-9
VIN_AMPLITUDE = 1.0
INPUT_FREQUENCY_HZ = 1_000.0
AC_START_HZ = 10.0
AC_STOP_HZ = 1_000_000.0
POINTS_PER_DECADE = 200

THEORY_TAU_S = R_OHM * C_FARAD
THEORY_FC_HZ = 1.0 / (2.0 * math.pi * THEORY_TAU_S)


@contextmanager
def ngspice_working_directory():
    """Run Ngspice from its package directory so spinit resolves correctly."""
    original_directory = Path.cwd()
    os.chdir(NgSpiceShared.NGSPICE_PATH)
    try:
        yield
    finally:
        os.chdir(original_directory)


def _as_float_array(waveform) -> np.ndarray:
    return np.asarray(waveform, dtype=float).reshape(-1)


def build_circuit(name: str, mode: str) -> Circuit:
    """Build either the transient or AC form of the RC filter."""
    if mode not in {"transient", "ac"}:
        raise ValueError(f"Unsupported mode: {mode}")

    circuit = Circuit(name)
    if mode == "transient":
        circuit.PulseVoltageSource(
            "in",
            "vin",
            circuit.gnd,
            initial_value=0 @ u_V,
            pulsed_value=VIN_AMPLITUDE @ u_V,
            pulse_width=0.5 @ u_ms,
            period=1 @ u_ms,
            delay_time=0.5 @ u_ms,
            rise_time=1 @ u_ns,
            fall_time=1 @ u_ns,
        )
    else:
        circuit.SinusoidalVoltageSource(
            "in",
            "vin",
            circuit.gnd,
            amplitude=0 @ u_V,
            frequency=INPUT_FREQUENCY_HZ @ u_Hz,
            ac_magnitude=VIN_AMPLITUDE @ u_V,
        )

    circuit.R("1", "vin", "vout", (R_OHM / 1_000.0) @ u_kOhm)
    circuit.C("1", "vout", circuit.gnd, (C_FARAD / 1e-9) @ u_nF)
    return circuit


def _estimate_tau(time_s: np.ndarray, vout_v: np.ndarray) -> float:
    rising_start = 0.5e-3
    rising_end = 1.0e-3
    segment = (time_s >= rising_start) & (time_s < rising_end)
    t_segment = time_s[segment]
    v_segment = vout_v[segment]
    if t_segment.size < 20:
        raise ValueError("Not enough transient samples to estimate tau")

    normalized = v_segment / VIN_AMPLITUDE
    fit_mask = (normalized >= 0.05) & (normalized <= 0.95) & (normalized < 1.0)
    if np.count_nonzero(fit_mask) < 10:
        raise ValueError("Not enough samples for a stable tau fit")
    slope, _ = np.polyfit(t_segment[fit_mask], np.log1p(-normalized[fit_mask]), 1)
    if slope >= 0.0:
        raise ValueError("Invalid exponential fit slope")
    return -1.0 / slope


def _estimate_cutoff_frequency(frequency_hz: np.ndarray, magnitude: np.ndarray) -> float:
    target = 1.0 / math.sqrt(2.0)
    for index in range(frequency_hz.size - 1):
        if magnitude[index] >= target > magnitude[index + 1]:
            f0, f1 = frequency_hz[index], frequency_hz[index + 1]
            m0, m1 = magnitude[index], magnitude[index + 1]
            fraction = (target - m0) / (m1 - m0)
            return float(f0 + fraction * (f1 - f0))
    raise ValueError("The AC response never crossed the -3 dB level")


def run_simulation() -> dict:
    """Run transient and AC analyses and return numerical results."""
    with ngspice_working_directory():
        transient_circuit = build_circuit("RC low-pass transient", "transient")
        transient_analysis = transient_circuit.simulator(
            temperature=25, nominal_temperature=25
        ).transient(step_time=1 @ u_us, end_time=3 @ u_ms)

        ac_circuit = build_circuit("RC low-pass AC sweep", "ac")
        ac_analysis = ac_circuit.simulator(
            temperature=25, nominal_temperature=25
        ).ac(
            start_frequency=AC_START_HZ @ u_Hz,
            stop_frequency=AC_STOP_HZ @ u_Hz,
            number_of_points=POINTS_PER_DECADE,
            variation="dec",
        )

    time_s = _as_float_array(transient_analysis.time)
    vin_v = _as_float_array(transient_analysis["vin"])
    vout_v = _as_float_array(transient_analysis["vout"])
    frequency_hz = _as_float_array(ac_analysis.frequency)
    ac_vout = np.asarray(ac_analysis["vout"], dtype=complex).reshape(-1)
    magnitude = np.abs(ac_vout)
    phase_deg = np.unwrap(np.angle(ac_vout)) * 180.0 / math.pi

    tau_simulated = _estimate_tau(time_s, vout_v)
    fc_simulated = _estimate_cutoff_frequency(frequency_hz, magnitude)
    attenuation_at_1mhz = float(np.interp(1e6, frequency_hz, magnitude))

    return {
        "time_s": time_s,
        "vin_v": vin_v,
        "vout_v": vout_v,
        "frequency_hz": frequency_hz,
        "magnitude": magnitude,
        "phase_deg": phase_deg,
        "tau_simulated": tau_simulated,
        "fc_simulated": fc_simulated,
        "attenuation_at_1mhz": attenuation_at_1mhz,
    }


def _percent_error(simulated: float, theoretical: float) -> float:
    return abs(simulated - theoretical) / abs(theoretical) * 100.0


def verify_results(results: dict) -> dict:
    """Compare extracted simulation values with analytical values."""
    tau_error = _percent_error(results["tau_simulated"], THEORY_TAU_S)
    fc_error = _percent_error(results["fc_simulated"], THEORY_FC_HZ)
    checks = {
        "tau_within_5_percent": tau_error <= 5.0,
        "fc_within_5_percent": fc_error <= 5.0,
        "dc_gain_near_unity": results["magnitude"][0] > 0.99,
        "high_frequency_rolloff_present": results["attenuation_at_1mhz"] < 0.01,
    }
    return {
        "tau_error_percent": tau_error,
        "fc_error_percent": fc_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def plot_results(results: dict) -> None:
    """Write transient and Bode plots to the assets directory."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    time_ms = results["time_s"] * 1e3
    ax.plot(time_ms, results["vin_v"], color="#7b8794", linestyle="--", label="Input (V)")
    ax.plot(time_ms, results["vout_v"], color="#0e7490", linewidth=2.0, label="Output (V)")
    ax.set_title("RC Low-Pass Filter - Transient Response")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(-0.08, 1.12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(ASSETS_DIR / "rc_transient.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True, constrained_layout=True)
    axes[0].semilogx(
        results["frequency_hz"], 20.0 * np.log10(results["magnitude"]), color="#0e7490", linewidth=2.0
    )
    axes[0].axhline(-3.0103, color="#b45309", linestyle="--", linewidth=1.2, label="-3 dB")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("RC Low-Pass Filter - AC Frequency Response")
    axes[0].grid(True, which="both", alpha=0.22)
    axes[0].legend(loc="upper right")

    axes[1].semilogx(results["frequency_hz"], results["phase_deg"], color="#1d4ed8", linewidth=2.0)
    axes[1].axhline(-45.0, color="#b45309", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].grid(True, which="both", alpha=0.22)
    fig.savefig(ASSETS_DIR / "rc_bode.png", dpi=180)
    plt.close(fig)


def write_circuit_svg() -> None:
    """Write a self-contained, editable SVG of the RC low-pass circuit."""
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 430" role="img" aria-labelledby="title desc">
  <title id="title">RC low-pass filter circuit</title>
  <desc id="desc">Voltage source, 1 kilohm resistor and 100 nanofarad capacitor forming a low-pass filter.</desc>
  <rect width="900" height="430" fill="#ffffff"/>
  <g fill="none" stroke="#172554" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="130" cy="285" r="34"/>
    <line x1="130" y1="319" x2="130" y2="365"/>
    <line x1="114" y1="330" x2="146" y2="330"/>
    <line x1="120" y1="341" x2="140" y2="341"/>
    <line x1="130" y1="251" x2="130" y2="120"/>
    <line x1="130" y1="120" x2="250" y2="120"/>
    <rect x="250" y="99" width="130" height="42" rx="4"/>
    <line x1="380" y1="120" x2="560" y2="120"/>
    <line x1="560" y1="120" x2="560" y2="198"/>
    <line x1="526" y1="198" x2="594" y2="198"/>
    <line x1="560" y1="218" x2="560" y2="238"/>
    <line x1="560" y1="238" x2="560" y2="365"/>
    <line x1="542" y1="365" x2="578" y2="365"/>
    <line x1="548" y1="375" x2="572" y2="375"/>
    <line x1="554" y1="385" x2="566" y2="385"/>
    <line x1="560" y1="120" x2="790" y2="120"/>
    <line x1="790" y1="120" x2="790" y2="365"/>
    <line x1="772" y1="365" x2="808" y2="365"/>
  </g>
  <g fill="#172554" font-family="Arial, Helvetica, sans-serif" font-size="18">
    <text x="97" y="180">Vin</text>
    <text x="160" y="282">0-1 V</text>
    <text x="160" y="306">1 kHz</text>
    <text x="288" y="83">R = 1 kOhm</text>
    <text x="600" y="220">C = 100 nF</text>
    <text x="752" y="100">Vout</text>
    <text x="73" y="402">Ground</text>
  </g>
  <circle cx="560" cy="120" r="5" fill="#0e7490"/>
  <circle cx="790" cy="120" r="5" fill="#0e7490"/>
</svg>
"""
    (ASSETS_DIR / "rc_circuit.svg").write_text(svg, encoding="utf-8")


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_simulation()
    verification = verify_results(results)
    plot_results(results)
    write_circuit_svg()

    print("RC Low-Pass Filter")
    print(f"Theory: tau={THEORY_TAU_S * 1e6:.3f} us, fc={THEORY_FC_HZ:.2f} Hz")
    print(
        "Simulation: "
        f"tau={results['tau_simulated'] * 1e6:.3f} us, "
        f"fc={results['fc_simulated']:.2f} Hz, "
        f"|H(1MHz)|={results['attenuation_at_1mhz']:.6f}"
    )
    print(
        "Error: "
        f"tau={verification['tau_error_percent']:.3f}%, "
        f"fc={verification['fc_error_percent']:.3f}%"
    )
    for name, passed in verification["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"RESULT: {'PASS' if verification['passed'] else 'FAIL'}")
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
