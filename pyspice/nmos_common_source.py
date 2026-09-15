"""NMOS common-source amplifier simulation with PySpice.

The circuit uses a Level 1 NMOS model with the task parameters. Both the ideal
square-law values and the channel-length-modulation-corrected values are
reported so the comparison remains physically explicit.
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
from PySpice.Unit import u_uF, u_Hz, u_kOhm, u_ms, u_us, u_um, u_V


SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"

VDD_V = 5.0
RG1_OHM = 60_000.0
RG2_OHM = 40_000.0
RD_OHM = 2_000.0
K_A_PER_V2 = 0.8e-3
VTH_V = 1.0
LAMBDA_PER_V = 0.02
CB1_FARAD = 10e-6
VIN_AMPLITUDE_V = 10e-3
INPUT_FREQUENCY_HZ = 1_000.0
VGS_BIAS_THEORY_V = VDD_V * RG2_OHM / (RG1_OHM + RG2_OHM)
GM_TEST_DELTA_V = 1e-3

IDEAL_ID_A = K_A_PER_V2 * (VGS_BIAS_THEORY_V - VTH_V) ** 2
IDEAL_VDS_V = VDD_V - IDEAL_ID_A * RD_OHM
IDEAL_GM_S = 2.0 * K_A_PER_V2 * (VGS_BIAS_THEORY_V - VTH_V)
IDEAL_RO_OHM = 1.0 / (LAMBDA_PER_V * IDEAL_ID_A)
IDEAL_AV = -IDEAL_GM_S * (RD_OHM * IDEAL_RO_OHM / (RD_OHM + IDEAL_RO_OHM))

CORRECTED_ID_A = (
    IDEAL_ID_A
    * (1.0 + LAMBDA_PER_V * VDD_V)
    / (1.0 + LAMBDA_PER_V * IDEAL_ID_A * RD_OHM)
)
CORRECTED_VDS_V = VDD_V - CORRECTED_ID_A * RD_OHM
CORRECTED_GM_S = (
    2.0
    * K_A_PER_V2
    * (VGS_BIAS_THEORY_V - VTH_V)
    * (1.0 + LAMBDA_PER_V * CORRECTED_VDS_V)
)
CORRECTED_RO_OHM = 1.0 / (LAMBDA_PER_V * CORRECTED_ID_A)
CORRECTED_AV = -CORRECTED_GM_S * (
    RD_OHM * CORRECTED_RO_OHM / (RD_OHM + CORRECTED_RO_OHM)
)


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


def _add_nmos_model(circuit: Circuit) -> None:
    # The task parameter K is 0.8 mA/V^2. SPICE Level 1 KP equals 2K,
    # therefore kp=1.6e-3 is the correct model value when W/L=1.
    circuit.model(
        "NMOS_MODEL",
        "nmos",
        level=1,
        vto=VTH_V,
        kp=2.0 * K_A_PER_V2,
        lambda_=LAMBDA_PER_V,
    )


def build_circuit(name: str = "NMOS common-source amplifier", mode: str = "transient") -> Circuit:
    """Build the complete common-source amplifier."""
    if mode not in {"operating_point", "transient"}:
        raise ValueError(f"Unsupported mode: {mode}")

    circuit = Circuit(name)
    circuit.V("DD", "vdd", circuit.gnd, VDD_V @ u_V)
    circuit.R("G1", "vdd", "gate", (RG1_OHM / 1_000.0) @ u_kOhm)
    circuit.R("G2", "gate", circuit.gnd, (RG2_OHM / 1_000.0) @ u_kOhm)
    circuit.R("D", "vdd", "drain", (RD_OHM / 1_000.0) @ u_kOhm)
    circuit.C("B1", "vin", "gate", (CB1_FARAD / 1e-6) @ u_uF)
    circuit.SinusoidalVoltageSource(
        "in",
        "vin",
        circuit.gnd,
        amplitude=VIN_AMPLITUDE_V @ u_V if mode == "transient" else 0 @ u_V,
        frequency=INPUT_FREQUENCY_HZ @ u_Hz,
    )
    _add_nmos_model(circuit)
    circuit.MOSFET(
        "1",
        "drain",
        "gate",
        circuit.gnd,
        circuit.gnd,
        model="NMOS_MODEL",
        l=1 @ u_um,
        w=1 @ u_um,
    )
    return circuit


def _node_value(analysis, name: str) -> float:
    return float(np.asarray(analysis[name]).reshape(-1)[0])


def _simulate_current_at_bias(vgs_v: float, vds_v: float) -> float:
    """Measure drain current with both MOSFET terminal voltages fixed."""
    circuit = Circuit(f"NMOS gm bias {vgs_v:.6f} V")
    circuit.V("G", "gate", circuit.gnd, vgs_v @ u_V)
    circuit.V("D", "drain", circuit.gnd, vds_v @ u_V)
    _add_nmos_model(circuit)
    circuit.MOSFET(
        "1",
        "drain",
        "gate",
        circuit.gnd,
        circuit.gnd,
        model="NMOS_MODEL",
        l=1 @ u_um,
        w=1 @ u_um,
    )
    analysis = circuit.simulator(temperature=25, nominal_temperature=25).operating_point()
    # The voltage source current is negative for current flowing into its
    # positive terminal, so the drain current is the negated branch current.
    return -float(np.asarray(analysis.branches["vd"]).reshape(-1)[0])


def run_simulation() -> dict:
    """Run DC, finite-difference gm and transient simulations."""
    with ngspice_working_directory():
        operating_circuit = build_circuit("NMOS operating point", "operating_point")
        operating_analysis = operating_circuit.simulator(
            temperature=25, nominal_temperature=25
        ).operating_point()
        vgs_q = _node_value(operating_analysis, "gate")
        vds_q = _node_value(operating_analysis, "drain")
        id_q = (VDD_V - vds_q) / RD_OHM

        current_low = _simulate_current_at_bias(vgs_q - GM_TEST_DELTA_V, vds_q)
        current_high = _simulate_current_at_bias(vgs_q + GM_TEST_DELTA_V, vds_q)
        gm_s = (current_high - current_low) / (2.0 * GM_TEST_DELTA_V)

        transient_circuit = build_circuit("NMOS transient", "transient")
        transient_analysis = transient_circuit.simulator(
            temperature=25, nominal_temperature=25
        ).transient(step_time=5 @ u_us, end_time=5 @ u_ms)

    time_s = _as_float_array(transient_analysis.time)
    vin_v = _as_float_array(transient_analysis["vin"])
    vout_v = _as_float_array(transient_analysis["drain"])
    settled = time_s >= 4.0e-3
    vin_settled = vin_v[settled]
    vout_settled = vout_v[settled]
    if vin_settled.size < 20:
        raise ValueError("Not enough settled transient samples")

    fit_matrix = np.column_stack((vin_settled, np.ones_like(vin_settled)))
    slope, intercept = np.linalg.lstsq(fit_matrix, vout_settled, rcond=None)[0]
    fitted_output = fit_matrix @ np.array([slope, intercept])
    residual_rms = float(np.sqrt(np.mean((vout_settled - fitted_output) ** 2)))
    output_swing = float(np.max(vout_settled) - np.min(vout_settled))
    input_swing = float(np.max(vin_settled) - np.min(vin_settled))
    correlation = float(np.corrcoef(vin_settled, vout_settled)[0, 1])

    return {
        "time_s": time_s,
        "vin_v": vin_v,
        "vout_v": vout_v,
        "vin_settled_v": vin_settled,
        "vout_settled_v": vout_settled,
        "vgs_q_v": vgs_q,
        "vds_q_v": vds_q,
        "id_q_a": id_q,
        "gm_s": gm_s,
        "gain": -slope,
        "signed_gain": float(slope),
        "input_peak_to_peak_v": input_swing,
        "output_peak_to_peak_v": output_swing,
        "distortion_residual_percent": residual_rms / max(output_swing, 1e-15) * 100.0,
        "correlation": correlation,
    }


def _error_percent(simulated: float, theoretical: float) -> float:
    return abs(simulated - theoretical) / max(abs(theoretical), 1e-15) * 100.0


def verify_results(results: dict) -> dict:
    """Verify Q point, small-signal parameters and transient behaviour."""
    corrected_errors = {
        "vgs_error_percent": _error_percent(results["vgs_q_v"], VGS_BIAS_THEORY_V),
        "id_error_percent": _error_percent(results["id_q_a"], CORRECTED_ID_A),
        "vds_error_percent": _error_percent(results["vds_q_v"], CORRECTED_VDS_V),
        "gm_error_percent": _error_percent(results["gm_s"], CORRECTED_GM_S),
        "gain_error_percent": _error_percent(results["gain"], abs(CORRECTED_AV)),
    }
    ideal_reference_errors = {
        "id_error_vs_ideal_percent": _error_percent(results["id_q_a"], IDEAL_ID_A),
        "vds_error_vs_ideal_percent": _error_percent(results["vds_q_v"], IDEAL_VDS_V),
        "gm_error_vs_ideal_percent": _error_percent(results["gm_s"], IDEAL_GM_S),
        "gain_error_vs_ideal_percent": _error_percent(results["gain"], abs(IDEAL_AV)),
    }
    checks = {
        "vgs_within_5_percent": corrected_errors["vgs_error_percent"] <= 5.0,
        "id_within_5_percent": corrected_errors["id_error_percent"] <= 5.0,
        "vds_within_5_percent": corrected_errors["vds_error_percent"] <= 5.0,
        "gm_within_5_percent": corrected_errors["gm_error_percent"] <= 5.0,
        "gain_within_10_percent": corrected_errors["gain_error_percent"] <= 10.0,
        "output_is_inverted": results["signed_gain"] < 0.0 and results["correlation"] < -0.99,
        "waveform_is_linear": results["distortion_residual_percent"] < 5.0,
        "output_does_not_clip": 0.1 < float(np.min(results["vout_settled_v"])) and float(
            np.max(results["vout_settled_v"])
        ) < VDD_V - 0.1,
    }
    return {
        "corrected_errors": corrected_errors,
        "ideal_reference_errors": ideal_reference_errors,
        "checks": checks,
        "passed": all(checks.values()),
    }


def plot_results(results: dict) -> None:
    """Plot input and output transient waveforms after DC settling."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    mask = results["time_s"] >= 4.0e-3
    time_ms = results["time_s"][mask] * 1e3
    input_mv = results["vin_v"][mask] * 1e3
    output_v = results["vout_v"][mask]

    figure, axes = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, input_mv, color="#0e7490", linewidth=2.0)
    axes[0].set_ylabel("Input (mV)")
    axes[0].set_title("NMOS Common-Source Amplifier - Transient Response")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time_ms, output_v, color="#b45309", linewidth=2.0)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Output (V)")
    axes[1].grid(True, alpha=0.25)
    figure.savefig(ASSETS_DIR / "nmos_waveforms.png", dpi=180)
    plt.close(figure)


def _full_circuit_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 650" role="img" aria-labelledby="title desc">
  <title id="title">NMOS common-source amplifier</title>
  <desc id="desc">Common-source amplifier with a resistive divider, input coupling capacitor, drain resistor and Level 1 NMOS transistor.</desc>
  <rect width="960" height="650" fill="#ffffff"/>
  <g fill="none" stroke="#172554" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <line x1="180" y1="70" x2="780" y2="70"/><line x1="300" y1="70" x2="300" y2="130"/><rect x="278" y="130" width="44" height="110" rx="4"/>
    <line x1="300" y1="240" x2="300" y2="420"/><line x1="300" y1="420" x2="300" y2="490"/><rect x="278" y="490" width="44" height="95" rx="4"/><line x1="300" y1="585" x2="300" y2="610"/>
    <line x1="278" y1="610" x2="322" y2="610"/><line x1="287" y1="620" x2="313" y2="620"/><line x1="294" y1="630" x2="306" y2="630"/>
    <line x1="300" y1="360" x2="390" y2="360"/><line x1="390" y1="330" x2="390" y2="450"/>
    <line x1="120" y1="360" x2="210" y2="360"/><line x1="210" y1="330" x2="210" y2="390"/><line x1="230" y1="330" x2="230" y2="390"/><line x1="230" y1="360" x2="300" y2="360"/>
    <circle cx="120" cy="460" r="32"/><line x1="108" y1="452" x2="132" y2="468"/><line x1="120" y1="492" x2="120" y2="610"/><line x1="98" y1="610" x2="142" y2="610"/><line x1="107" y1="620" x2="133" y2="620"/><line x1="114" y1="630" x2="126" y2="630"/><line x1="120" y1="360" x2="120" y2="428"/>
    <line x1="580" y1="70" x2="580" y2="145"/><rect x="558" y="145" width="44" height="110" rx="4"/><line x1="580" y1="255" x2="580" y2="330"/><line x1="580" y1="330" x2="580" y2="450"/>
    <line x1="390" y1="360" x2="470" y2="360"/><line x1="470" y1="325" x2="470" y2="395"/><line x1="510" y1="310" x2="510" y2="410"/><line x1="510" y1="330" x2="580" y2="330"/><line x1="510" y1="350" x2="550" y2="350"/><path d="M535 350 L550 350 L542 338 Z" fill="#0e7490" stroke="#0e7490"/><line x1="550" y1="350" x2="580" y2="350"/><line x1="510" y1="390" x2="580" y2="390"/><line x1="510" y1="410" x2="550" y2="410"/><path d="M530 410 L510 410 L522 398 Z" fill="#0e7490" stroke="#0e7490"/><line x1="550" y1="410" x2="580" y2="410"/>
    <line x1="580" y1="450" x2="580" y2="540"/><line x1="560" y1="540" x2="600" y2="540"/><line x1="568" y1="550" x2="592" y2="550"/><line x1="574" y1="560" x2="586" y2="560"/>
    <line x1="580" y1="330" x2="760" y2="330"/><line x1="760" y1="330" x2="760" y2="380"/><circle cx="760" cy="410" r="30"/><line x1="748" y1="402" x2="772" y2="418"/><line x1="760" y1="440" x2="760" y2="540"/><line x1="738" y1="540" x2="782" y2="540"/><line x1="747" y1="550" x2="773" y2="550"/><line x1="754" y1="560" x2="766" y2="560"/>
    <line x1="680" y1="330" x2="680" y2="275"/><line x1="660" y1="275" x2="700" y2="275"/><line x1="668" y1="265" x2="692" y2="265"/><line x1="674" y1="255" x2="686" y2="255"/>
  </g>
  <g fill="#172554" font-family="Arial, Helvetica, sans-serif" font-size="18">
    <text x="195" y="48">VDD = 5 V</text><text x="165" y="180">Rg1 = 60 kOhm</text><text x="165" y="546">Rg2 = 40 kOhm</text>
    <text x="65" y="340">Vin</text><text x="55" y="525">10 mV, 1 kHz</text><text x="185" y="320">Cb1 = 10 uF</text>
    <text x="615" y="205">Rd = 2 kOhm</text><text x="600" y="315">Vout</text><text x="385" y="290">Gate</text><text x="430" y="490">M1 (Level 1)</text>
  </g>
  <circle cx="300" cy="70" r="5" fill="#0e7490"/><circle cx="580" cy="70" r="5" fill="#0e7490"/><circle cx="580" cy="330" r="5" fill="#0e7490"/>
</svg>
"""


def _dc_path_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520" role="img" aria-labelledby="title desc">
  <title id="title">NMOS common-source DC bias path</title>
  <desc id="desc">DC path with a five volt supply, sixty and forty kilohm gate divider, two kilohm drain resistor and NMOS transistor.</desc>
  <rect width="900" height="520" fill="#ffffff"/>
  <g fill="none" stroke="#172554" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <line x1="140" y1="70" x2="760" y2="70"/><line x1="290" y1="70" x2="290" y2="135"/><rect x="268" y="135" width="44" height="115" rx="4"/><line x1="290" y1="250" x2="290" y2="390"/><rect x="268" y="390" width="44" height="105" rx="4"/><line x1="290" y1="495" x2="290" y2="515"/><line x1="270" y1="515" x2="310" y2="515"/>
    <line x1="290" y1="305" x2="420" y2="305"/><line x1="420" y1="275" x2="420" y2="405"/><line x1="460" y1="260" x2="460" y2="420"/><line x1="460" y1="305" x2="610" y2="305"/><line x1="460" y1="330" x2="585" y2="330"/><path d="M565 330 L585 330 L576 318 Z" fill="#0e7490" stroke="#0e7490"/><line x1="585" y1="330" x2="610" y2="330"/><line x1="460" y1="390" x2="610" y2="390"/><line x1="610" y1="305" x2="610" y2="390"/><line x1="610" y1="390" x2="610" y2="515"/><line x1="590" y1="515" x2="630" y2="515"/>
    <line x1="610" y1="70" x2="610" y2="135"/><rect x="588" y="135" width="44" height="115" rx="4"/><line x1="610" y1="250" x2="610" y2="305"/>
  </g>
  <g fill="#172554" font-family="Arial, Helvetica, sans-serif" font-size="18">
    <text x="155" y="47">VDD = 5 V</text><text x="105" y="195">Rg1 = 60 kOhm</text><text x="105" y="452">Rg2 = 40 kOhm</text><text x="635" y="200">Rd = 2 kOhm</text>
    <text x="330" y="280">VGS = 2 V</text><text x="635" y="325">VDS ~= 3.29 V</text><text x="635" y="370">ID ~= 0.853 mA</text><text x="405" y="470">M1, Level 1</text>
  </g>
  <circle cx="290" cy="70" r="5" fill="#0e7490"/><circle cx="610" cy="70" r="5" fill="#0e7490"/><circle cx="610" cy="305" r="5" fill="#0e7490"/>
</svg>
"""


def _small_signal_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520" role="img" aria-labelledby="title desc">
  <title id="title">NMOS common-source small-signal model</title>
  <desc id="desc">Small-signal equivalent circuit with a gate bias resistance, transconductance source, output resistance and drain resistor.</desc>
  <rect width="900" height="520" fill="#ffffff"/>
  <g fill="none" stroke="#172554" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="120" cy="260" r="32"/><line x1="108" y1="252" x2="132" y2="268"/><line x1="120" y1="292" x2="120" y2="440"/>
    <line x1="120" y1="260" x2="300" y2="260"/><line x1="300" y1="260" x2="300" y2="390"/><line x1="300" y1="390" x2="300" y2="440"/>
    <line x1="120" y1="440" x2="680" y2="440"/><line x1="300" y1="390" x2="390" y2="390"/><line x1="390" y1="360" x2="390" y2="420"/>
    <line x1="430" y1="340" x2="430" y2="440"/><line x1="430" y1="390" x2="520" y2="390"/><line x1="430" y1="360" x2="520" y2="360"/>
    <line x1="520" y1="310" x2="660" y2="310"/><line x1="660" y1="310" x2="660" y2="440"/>
    <rect x="518" y="230" width="44" height="80" rx="4"/><line x1="520" y1="310" x2="520" y2="310"/>
    <line x1="520" y1="270" x2="660" y2="270"/><line x1="660" y1="270" x2="660" y2="310"/><rect x="638" y="190" width="44" height="80" rx="4"/>
    <path d="M520 360 L545 360 L560 360 M520 390 L560 390" stroke="#0e7490"/><line x1="560" y1="360" x2="560" y2="390" stroke="#0e7490"/><path d="M548 360 L560 342 L572 360 Z" fill="#0e7490" stroke="#0e7490"/><line x1="560" y1="390" x2="660" y2="390"/><line x1="660" y1="390" x2="660" y2="440"/>
    <line x1="660" y1="270" x2="660" y2="270" stroke="#0e7490"/><line x1="660" y1="190" x2="660" y2="130"/><line x1="660" y1="310" x2="800" y2="310"/><line x1="800" y1="310" x2="800" y2="440"/>
    <line x1="630" y1="130" x2="690" y2="130"/><line x1="638" y1="120" x2="682" y2="120"/>
  </g>
  <g fill="#172554" font-family="Arial, Helvetica, sans-serif" font-size="18">
    <text x="75" y="235">vs</text><text x="235" y="245">vgs</text><text x="320" y="365">Rg = Rg1 || Rg2</text><text x="475" y="330">gm vgs</text><text x="535" y="215">ro</text><text x="695" y="215">Rd</text><text x="735" y="290">vout</text><text x="150" y="485">Source / ground</text>
  </g>
  <circle cx="300" cy="260" r="5" fill="#0e7490"/><circle cx="660" cy="310" r="5" fill="#0e7490"/>
</svg>
"""


def write_circuit_svgs() -> None:
    (ASSETS_DIR / "nmos_circuit.svg").write_text(_full_circuit_svg(), encoding="utf-8")
    (ASSETS_DIR / "nmos_dc_path.svg").write_text(_dc_path_svg(), encoding="utf-8")
    (ASSETS_DIR / "nmos_small_signal.svg").write_text(_small_signal_svg(), encoding="utf-8")


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_simulation()
    verification = verify_results(results)
    plot_results(results)
    write_circuit_svgs()

    print("NMOS Common-Source Amplifier")
    print(
        "Ideal square-law (lambda=0): "
        f"VGS={VGS_BIAS_THEORY_V:.3f} V, ID={IDEAL_ID_A * 1e3:.4f} mA, "
        f"VDS={IDEAL_VDS_V:.4f} V, gm={IDEAL_GM_S * 1e3:.4f} mS, "
        f"ro={IDEAL_RO_OHM / 1e3:.4f} kOhm, Av={IDEAL_AV:.4f}"
    )
    print(
        "Lambda-corrected theory: "
        f"ID={CORRECTED_ID_A * 1e3:.4f} mA, VDS={CORRECTED_VDS_V:.4f} V, "
        f"gm={CORRECTED_GM_S * 1e3:.4f} mS, "
        f"ro={CORRECTED_RO_OHM / 1e3:.4f} kOhm, Av={CORRECTED_AV:.4f}"
    )
    print(
        "Simulation Q point: "
        f"VGS={results['vgs_q_v']:.6f} V, VDS={results['vds_q_v']:.6f} V, "
        f"ID={results['id_q_a'] * 1e3:.6f} mA"
    )
    print(
        "Simulation small signal: "
        f"gm={results['gm_s'] * 1e3:.6f} mS, measured gain={results['gain']:.6f}, "
        f"correlation={results['correlation']:.7f}"
    )
    print(
        "Waveform: "
        f"Vin_pp={results['input_peak_to_peak_v'] * 1e3:.6f} mV, "
        f"Vout_pp={results['output_peak_to_peak_v']:.6f} V, "
        f"residual={results['distortion_residual_percent']:.4f}%"
    )
    print("Errors vs lambda-corrected theory:")
    for name, value in verification["corrected_errors"].items():
        print(f"  {name}: {value:.4f}%")
    print("Informational errors vs ideal square-law theory:")
    for name, value in verification["ideal_reference_errors"].items():
        print(f"  {name}: {value:.4f}%")
    for name, passed in verification["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"RESULT: {'PASS' if verification['passed'] else 'FAIL'}")
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())



