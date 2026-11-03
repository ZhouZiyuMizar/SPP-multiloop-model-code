# -*- coding: utf-8 -*-
"""Model-only code for the SPP multi-loop study.

This module contains only the constitutive solvers, the SPP/Fourier analysis,
the three counting measures and the numerical checks. It does not contain any
plotting code, figure source data or manuscript figure files.

Python 3.8+; numpy and scipy only.

Commands:
    python source/spp_model.py check
    python source/spp_model.py cases
    python source/spp_model.py sweep
    python source/spp_model.py harmonics
"""
import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks


@dataclass
class Config:
    name: str = "two_groups"
    model: str = "evp"  # evp, recovery, or elastic
    G: tuple = (0.5, 0.5)  # Pa
    yield_strain: tuple = (0.20, 0.30)  # Ymax/G, dimensionless
    gamma0: float = 1.0
    omega: float = 1.0  # rad/s
    Ge: float = 0.1  # Pa
    eta_s: float = 0.02  # Pa s
    rate0: float = 20.0  # 1/s
    exponent: float = 3.0
    recovery_time: float = 1.0  # s
    breakdown: float = 1.0  # per unit plastic strain
    ymin_fraction: float = 0.35

    def validate(self):
        if self.model not in ("evp", "recovery", "elastic"):
            raise ValueError("model must be evp, recovery, or elastic")
        if len(self.G) != len(self.yield_strain) or not len(self.G):
            raise ValueError("G and yield_strain must have equal nonzero lengths")
        if min(self.G) <= 0 or min(self.yield_strain) <= 0:
            raise ValueError("G and yield_strain must be positive")
        if min(self.gamma0, self.omega, self.rate0, self.recovery_time) <= 0:
            raise ValueError("amplitude, frequency and kinetic scales must be positive")
        if self.exponent < 3 or min(self.Ge, self.eta_s, self.breakdown) < 0:
            raise ValueError("exponent >= 3; Ge, eta_s, breakdown >= 0")
        if not 0 < self.ymin_fraction <= 1:
            raise ValueError("0 < ymin_fraction <= 1")


def kinetics(q, lam, c):
    """Return plastic rate and its partial derivatives v_q and v_lambda."""
    ymax = np.asarray(c.G) * np.asarray(c.yield_strain)
    dy = ymax * (1 - c.ymin_fraction) if c.model == "recovery" else ymax * 0
    Y = ymax * c.ymin_fraction + dy * lam if c.model == "recovery" else ymax
    x = np.maximum(np.abs(q) / Y - 1, 0)
    v = c.rate0 * x ** c.exponent * np.sign(q)
    vq = c.rate0 * c.exponent * x ** (c.exponent - 1) / Y
    vl = -vq * np.abs(q) * dy / Y * np.sign(q)
    return v, vq, vl, np.broadcast_to(Y, q.shape)


def rhs(theta, state, c):
    m = len(c.G)
    q = state[:m]
    lam = state[m:] if c.model == "recovery" else np.ones(m)
    v, _, _, _ = kinetics(q, lam, c)
    q_theta = np.asarray(c.G) * (c.gamma0 * np.cos(theta) - v / c.omega)
    if c.model != "recovery":
        return q_theta
    ltheta = ((1 - lam) / c.recovery_time - c.breakdown * lam * np.abs(v)) / c.omega
    return np.r_[q_theta, ltheta]


def coefficients(theta, s, nmax):
    """s = Im(sum C_n exp(i*n*theta)); theta covers 2pi without endpoint."""
    ft = np.fft.rfft(s) / len(s)
    n = np.arange(min(nmax, len(ft) - 1) + 1)
    C = 2j * ft[n] * np.exp(-1j * n * theta[0])
    C[0] = 0  # SPP is insensitive to a constant stress offset
    return C


def spp_fourier(theta, C, cutoff=None):
    cutoff = len(C) - 1 if cutoff is None else min(cutoff, len(C) - 1)
    z = np.zeros_like(theta, dtype=complex)
    for n in range(1, cutoff + 1):
        z += (n * (n + 1) * C[n] * np.exp(1j * (n - 1) * theta)
              - n * (n - 1) * C[n].conjugate() * np.exp(-1j * (n + 1) * theta)) / 2
    return z


def crossings(z, tol=2e-7, area_tolerance=1e-5):
    """Sampled polygon transverse intersections; NOT an exact topology oracle.

    Returns an Euler face estimate only when all detected crossings are double.
    Cusps, tangent contacts, overlapping arcs and unresolved tiny loops need
    separate inspection. Run at multiple spatial resolutions.
    """
    z = np.asarray(z, complex)
    scale = max(np.ptp(z.real), np.ptp(z.imag))
    if scale < 1e-12:
        return {"crossings": 0, "raw_crossings": 0, "generic_faces": 0,
                "multiple_contact": False, "area_tolerance": area_tolerance}
    p = np.c_[z.real - z.real.mean(), z.imag - z.imag.mean()] / scale
    r = np.roll(p, -1, axis=0) - p
    pp = np.roll(p, -1, axis=0)
    areas = np.r_[0, np.cumsum((p[:, 0]*pp[:, 1] - p[:, 1]*pp[:, 0]) / 2)]
    hits = []
    visits = []
    arc_areas = []
    N = len(p)
    for i in range(N - 2):
        jj = np.arange(i + 2, N)
        if i == 0:
            jj = jj[jj != N - 1]
        if not len(jj):
            continue
        a, b = r[i], r[jj]
        d = p[jj] - p[i]
        den = a[0] * b[:, 1] - a[1] * b[:, 0]
        lengths = np.linalg.norm(a) * np.linalg.norm(b, axis=1)
        good = np.abs(den) > 1e-6 * lengths + 1e-18
        safe = np.where(good, den, 1)
        u = (d[:, 0] * b[:, 1] - d[:, 1] * b[:, 0]) / safe
        v = (d[:, 0] * a[1] - d[:, 1] * a[0]) / safe
        ok = good & (u >= -1e-10) & (u <= 1 + 1e-10) & (v >= -1e-10) & (v <= 1 + 1e-10)
        for j, ui, vi in zip(jj[ok], u[ok], v[ok]):
            point = p[i] + ui * a
            first, last = p[(i + 1) % N], p[j]
            arc = ((point[0]*first[1] - point[1]*first[0]) / 2
                   + areas[j] - areas[i + 1]
                   + (last[0]*point[1] - last[1]*point[0]) / 2)
            # Both closed subarcs must have resolved signed area. This is an
            # operational size filter, not polygonization into individual faces.
            min_arc = min(abs(arc), abs(areas[-1] - arc))
            pair = np.array([(i + ui) % N, (j + vi) % N]) / N
            old = next((k for k, h in enumerate(hits) if np.linalg.norm(point - h) < tol), None)
            if old is None:
                hits.append(point)
                visits.append(list(pair))
                arc_areas.append(min_arc)
            else:
                arc_areas[old] = max(arc_areas[old], min_arc)
                for value in pair:
                    dist = np.abs(np.asarray(visits[old]) - value)
                    if np.min(np.minimum(dist, 1 - dist)) > 2 / N:
                        visits[old].append(value)
    multiple = any(len(v) > 2 for v in visits)
    resolved = int(sum(a >= area_tolerance for a in arc_areas))
    return {"crossings": resolved, "raw_crossings": len(hits),
            "generic_faces": None if multiple or resolved != len(hits) else len(hits) + 1,
            "multiple_contact": multiple, "area_tolerance": area_tolerance}


def evaluate(c, npoints=4096, rtol=1e-8, max_cycles=60):
    c.validate()
    if npoints < 1024 or npoints % 4:
        raise ValueError("npoints must be >= 1024 and divisible by four")
    th = np.linspace(-np.pi / 2, 3 * np.pi / 2, npoints, endpoint=False)
    m = len(c.G)
    g, gt = c.gamma0 * np.sin(th), c.gamma0 * c.omega * np.cos(th)
    if c.model == "elastic":
        sig = 20 * g - 8 * g ** 3 + 3.2 * g ** 5
        sp = (20 - 24 * g ** 2 + 16 * g ** 4) * np.cos(th)
        spp = ((-48 * g + 64 * g ** 3) * c.gamma0 * np.cos(th) ** 2
               - (20 - 24 * g ** 2 + 16 * g ** 4) * np.sin(th))
        z = np.exp(-1j * th) * (sp - 1j * spp)
        q = v = np.zeros((npoints, m))
        lam = np.ones_like(q)
        Y = np.ones_like(q)
        sig_t = (20 - 24 * g ** 2 + 16 * g ** 4) * gt
        dW = sig_t * 0 + sig * gt
        diss = sig * 0
        residual, cyc = 0.0, 0
    else:
        state = np.r_[np.zeros(m), np.ones(m)] if c.model == "recovery" else np.zeros(m)
        scales = np.r_[np.asarray(c.G) * c.gamma0, np.ones(m)] if c.model == "recovery" else np.asarray(c.G) * c.gamma0
        residual = np.inf
        for cyc in range(1, max_cycles + 1):
            sol = solve_ivp(lambda t, y: rhs(t, y, c), (th[0], th[0] + 2 * np.pi), state,
                            method="Radau", rtol=rtol, atol=rtol * 0.01,
                            dense_output=True, max_step=0.04)
            if not sol.success:
                raise RuntimeError(sol.message)
            final = sol.y[:, -1]
            residual = float(np.max(np.abs(final - state) / np.maximum(scales, 1e-12)))
            state = final
            if cyc >= 3 and residual < 5e-7:
                break
        if residual >= 5e-7:
            raise RuntimeError(f"Periodic state not reached: residual={residual}")
        states = sol.sol(th).T
        q = states[:, :m]
        lam = states[:, m:] if c.model == "recovery" else np.ones_like(q)
        v, vq, vl, Y = kinetics(q, lam, c)
        qt = np.asarray(c.G) * (gt[:, None] - v)
        lt = ((1 - lam) / c.recovery_time - c.breakdown * lam * np.abs(v)) if c.model == "recovery" else lam * 0
        vt = vq * qt + vl * lt
        sig = c.Ge * g + c.eta_s * gt + q.sum(axis=1)
        sig_t = c.Ge * gt - c.eta_s * c.omega ** 2 * g + qt.sum(axis=1)
        sig_tt = (-c.Ge * c.omega ** 2 * g - c.eta_s * c.omega ** 2 * gt
                  + (np.asarray(c.G) * (-c.omega ** 2 * g[:, None] - vt)).sum(axis=1))
        z = np.exp(-1j * th) * (sig_t / (c.gamma0 * c.omega)
                                - 1j * sig_tt / (c.gamma0 * c.omega ** 2))
        diss = c.eta_s * gt ** 2 + (q * v).sum(axis=1)
        dW = c.Ge * g * gt + (q * qt / np.asarray(c.G)).sum(axis=1)
    C = coefficients(th, sig / c.gamma0, min(511, npoints // 2 - 1))
    half = npoints // 2
    # Yield onset is defined by q_j - Y_j crossing from nonpositive to positive
    # during the increasing-strain half-cycle.
    # Residual negative plastic flow inherited from the previous half is excluded.
    events = []
    if c.model != "elastic":
        for j in range(m):
            f = q[:half + 1, j] - Y[:half + 1, j]
            for k in np.flatnonzero((f[:-1] <= 0) & (f[1:] > 0)):
                phase = th[k] - f[k] * (th[k + 1] - th[k]) / (f[k + 1] - f[k])
                events.append({"branch": j + 1, "phase_rad": float(phase)})
    events.sort(key=lambda e: e["phase_rad"])
    # Stiffness-weighted equivalent positive plastic strain rate.  Dividing by
    # gamma0*omega gives the dimensionless plastic-flow measure Phi_p^+ used in
    # the manuscript; constant scaling does not change peak locations/counts.
    equivalent_plastic_rate = (
        (np.asarray(c.G) * np.maximum(v[:half], 0)).sum(axis=1)
        / np.sum(c.G)
    )
    plastic_flow_measure = equivalent_plastic_rate / (c.gamma0 * c.omega)
    peaks, _ = find_peaks(plastic_flow_measure,
                          prominence=max(plastic_flow_measure.max() * 0.05, 1e-12),
                         distance=max(1, half // 100))
    work = 2 * np.pi / c.omega * np.mean(sig * gt)
    denom = max(abs(C[3]), 1e-14)
    summary = {"name": c.name, "model": c.model, "groups": 0 if c.model == "elastic" else m,
               "N_onset": len(events), "N_flow": len(peaks),
               # Legacy field names retained so existing result readers continue to work.
               "activation_count": len(events), "activity_peaks_5pct": len(peaks),
               "min_event_gap_rad": min(np.diff([e["phase_rad"] for e in events]), default=None),
               "cycles": cyc, "periodic_residual": residual,
               "work_per_cycle_Pa": float(work), "min_dissipation_Pa_per_s": float(diss.min()),
               "energy_identity_max_error": float(np.max(np.abs(sig * gt - dW - diss))),
               "half_period_Z_error": float(np.max(np.abs(z[:half] - z[half:]))),
               "C3_abs_Pa": float(abs(C[3])),
               "R5": float(5 * abs(C[5]) / denom), "R7": float(14 * abs(C[7]) / denom),
               "R9": float(30 * abs(C[9]) / denom),
               "C3_reference_valid": bool(abs(C[3]) > 1e-10 * max(abs(C[1]), 1e-12))}
    return {"config": c, "theta": th, "gamma": g, "stress": sig, "Z": z,
            "q": q, "v": v, "lambda": lam, "Y": Y, "dissipation": diss,
            "equivalent_plastic_rate": equivalent_plastic_rate,
            "plastic_flow_measure": plastic_flow_measure,
            "C": C, "events": events, "summary": summary}


def geometry_comparison(data):
    th, z, C = data["theta"], data["Z"], data["C"]
    half = len(th) // 2
    # Shift the sampled polygon off grid-aligned symmetric crossings.
    def sampled(source, n):
        t = th[0] + (np.arange(n) + 0.371) * np.pi / n
        xp = th[:half + 1]
        assert np.all(np.diff(xp) > 0), "interpolation phase grid must be strictly increasing"
        return np.interp(t, xp, source[:half + 1].real) + 1j * np.interp(t, xp, source[:half + 1].imag)
    rows = []
    for n in (256, 512, 1024):
        rows.append({"source": "ODE", "polygon_points": n, "cutoff": 0,
                     "Z_max_error_Pa": 0, **crossings(sampled(z, n))})
    for cut in (3, 5, 7, 9, 15, 31, 63, 127, 255, 511):
        approx = spp_fourier(th, C, cut)
        rows.append({"source": "Fourier", "polygon_points": 1024, "cutoff": cut,
                     "Z_max_error_Pa": float(np.max(np.abs(approx - z))),
                     **crossings(sampled(approx, 1024))})
    data["geometry"] = rows
    data["summary"]["ODE_crossings_1024"] = rows[2]["crossings"]
    data["summary"]["N_cross"] = rows[2]["crossings"]
    data["summary"]["ODE_raw_crossings_1024"] = rows[2]["raw_crossings"]
    for eps in (1e-6, 1e-4):
        data["summary"][f"ODE_crossings_area_{eps:g}"] = crossings(sampled(z, 1024), area_tolerance=eps)["crossings"]
    data["summary"]["crossings_sampling_stable"] = len(set(r["crossings"] for r in rows[:3])) == 1
    data["summary"]["Z_error_cutoff_511_Pa"] = rows[-1]["Z_max_error_Pa"]
    return rows


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_case(d, root):
    """Save one model calculation as CSV/JSON. No plotting is performed."""
    root = Path(root) / d["config"].name
    root.mkdir(parents=True, exist_ok=True)
    geometry_comparison(d)
    metadata = {
        "parameters": asdict(d["config"]),
        "summary": d["summary"],
        "events": d["events"],
        "notes": "N_onset, N_flow and N_cross describe constitutive yield-onset events, "
                 "resolvable macroscopic plastic-flow stages and resolvable SPP self-intersections.",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(root / "geometry_cutoff.csv", d["geometry"])
    write_csv(root / "harmonics.csv", ({"n": n, "Gprime_Pa": z.real, "Gdoubleprime_Pa": z.imag,
                                        "amplitude_Pa": abs(z), "phase_rad": np.angle(z)}
                                       for n, z in enumerate(d["C"]) if n > 0))
    columns = {"theta_rad": d["theta"], "gamma": d["gamma"], "stress_Pa": d["stress"],
               "Gt_prime_Pa": d["Z"].real, "Gt_doubleprime_Pa": d["Z"].imag,
               "dissipation_Pa_per_s": d["dissipation"]}
    for j in range(len(d["config"].G)):
        for key in ("q", "v", "lambda", "Y"):
            columns[f"{key}_{j+1}"] = d[key][:, j]
    write_csv(root / "waveform.csv", (dict(zip(columns, vals)) for vals in zip(*columns.values())))
    return d["summary"]


def manuscript_configs():
    """The six model configurations used in the manuscript."""
    return [
        Config(name="groups_1", model="evp", G=(1.0,), yield_strain=(0.42,)),
        Config(name="groups_2", model="evp", G=(0.5, 0.5), yield_strain=(0.16, 0.76)),
        Config(name="groups_3", model="evp", G=(0.25, 0.35, 0.40), yield_strain=(0.16, 0.26, 0.36)),
        Config(name="groups_4", model="evp", G=(0.25, 0.25, 0.25, 0.25),
               yield_strain=(0.16, 0.36, 0.56, 0.76)),
        Config(name="recovery_two", model="recovery", G=(0.5, 0.5), yield_strain=(0.20, 0.30),
               recovery_time=1.0, breakdown=1.0, ymin_fraction=0.35),
        Config(name="elastic_control", model="elastic"),
    ]


def config_from_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return Config(**data)


def single_case(root, config_path):
    d = evaluate(config_from_json(config_path))
    save_case(d, root)
    print(json.dumps(d["summary"]), flush=True)


def cases(root):
    rows, flow_rows = [], []
    for c in manuscript_configs():
        d = evaluate(c)
        rows.append(save_case(d, root))
        if c.model != "elastic":
            signal = d["plastic_flow_measure"]
            for prominence_fraction in (.02, .05, .10):
                for distance_fraction in (.005, .01, .02):
                    count = len(find_peaks(
                        signal,
                        prominence=max(signal.max() * prominence_fraction, 1e-12),
                        distance=max(1, int(len(signal) * distance_fraction)),
                    )[0])
                    flow_rows.append({
                        "name": c.name,
                        "prominence_fraction": prominence_fraction,
                        "halfcycle_distance_fraction": distance_fraction,
                        "minimum_peak_gap_deg": 180 * distance_fraction,
                        "N_flow": count,
                    })
        print(json.dumps(rows[-1]), flush=True)
    write_csv(Path(root) / "cases_summary.csv", rows)
    write_csv(Path(root) / "flow_peak_sensitivity.csv", flow_rows)


def sweep(root):
    rows = []
    for gap in (.05, .15, .30, .45, .60):
        for weight in (.2, .5, .8):
            c = Config(name=f"gap_{gap:.2f}_weight_{weight:.1f}", G=(1 - weight, weight),
                       yield_strain=(.15, .15 + gap))
            d = evaluate(c)
            geometry_comparison(d)
            row = {"yield_strain_gap": gap, "second_group_weight": weight,
                   "parameters_json": json.dumps(asdict(c)), **d["summary"]}
            for n in (3, 5, 7, 9):
                row[f"C{n}_real"] = float(d["C"][n].real)
                row[f"C{n}_imag"] = float(d["C"][n].imag)
            rows.append(row)
            print(c.name, row["activation_count"], row.get("ODE_crossings_1024"), flush=True)
    write_csv(Path(root) / "parameter_sweep.csv", rows)


def write_transitions(rows, root):
    transitions = []
    for a, b in zip(rows[:-1], rows[1:]):
        if (int(a["order"]) == int(b["order"]) and float(a["phase_rad"]) == float(b["phase_rad"])
                and int(a["crossings"]) != int(b["crossings"])):
            transitions.append({"order": int(a["order"]), "phase_rad": float(a["phase_rad"]),
                                "ratio_left": float(a["amplitude_ratio_to_C3"]),
                                "ratio_right": float(b["amplitude_ratio_to_C3"]),
                                "crossings_left": int(a["crossings"]), "crossings_right": int(b["crossings"]),
                                "status": "coarse_grid_bracket_requires_refinement"})
    write_csv(Path(root) / "harmonic_transition_brackets.csv", transitions)


def _exact_threshold(phase):
    theta = np.linspace(-np.pi / 2, np.pi / 2, 4001)

    def compatibility(value):
        return np.sin(8 * value + phase) + 4 * np.sin(2 * value + phase)

    values = compatibility(theta)
    roots = []
    for i in range(len(theta) - 1):
        if abs(values[i]) < 1e-11:
            roots.append(theta[i])
        if values[i] * values[i + 1] < 0:
            roots.append(brentq(compatibility, theta[i], theta[i + 1]))
    candidates = []
    for value in roots:
        ratio = np.sqrt(9 + 16 * np.cos(3 * value) ** 2) / 25
        first = np.cos(3 * value) + 5 * ratio * np.cos(5 * value + phase)
        second = 3 * np.sin(3 * value) + 25 * ratio * np.sin(5 * value + phase)
        if max(abs(first), abs(second)) < 1e-7:
            candidates.append(ratio)
    if not candidates:
        raise RuntimeError(f"No critical point found at phase {phase}")
    return min(candidates)


def _operational_threshold(phase, polygon_points=512, area_tolerance=1e-5, iterations=14):
    theta = -np.pi / 2 + (np.arange(polygon_points) + .371) * np.pi / polygon_points
    low, high = 0.0, 0.30
    for _ in range(iterations):
        ratio = (low + high) / 2
        C = np.zeros(6, complex)
        C[3], C[5] = 1, ratio * np.exp(1j * phase)
        resolved = crossings(spp_fourier(theta, C), area_tolerance=area_tolerance)["crossings"] > 0
        if resolved:
            high = ratio
        else:
            low = ratio
    return (low + high) / 2


def fifth_harmonic_thresholds(root):
    root = Path(root)
    phases = np.linspace(-np.pi, np.pi, 65)
    exact = np.array([_exact_threshold(p) for p in phases])
    operational = np.array([_operational_threshold(p) for p in phases])
    write_csv(root / "fifth_harmonic_thresholds.csv", [{
        "phase_rad": p, "phase_deg": np.degrees(p),
        "exact_ratio_threshold": e, "resolved_ratio_threshold": o,
        "polygon_points": 512, "area_tolerance": 1e-5,
    } for p, e, o in zip(phases, exact, operational)])

    step = phases[1] - phases[0]
    lo = int(np.argmin(operational)); hi = int(np.argmax(operational))

    def refine(index, sign):
        lower = max(-np.pi, phases[index] - step)
        upper = min(np.pi, phases[index] + step)
        result = minimize_scalar(lambda v: sign * _operational_threshold(v),
                                 bounds=(lower, upper), method="bounded",
                                 options={"xatol": 2e-4})
        p = float(result.x)
        return p, _operational_threshold(p), _operational_threshold(p, polygon_points=1024)

    min_phase, min_512, min_1024 = refine(lo, 1)
    max_phase, max_512, max_1024 = refine(hi, -1)
    summary = [
        {"criterion": "exact_no_phase_below", "ratio": 3 / 25, "R5": 3 / 5},
        {"criterion": "exact_all_phases_above", "ratio": 1 / 5, "R5": 1},
        {"criterion": "resolved_no_sampled_phase_below", "ratio": min_1024, "R5": 5 * min_1024,
         "phase_rad": min_phase, "phase_deg": np.degrees(min_phase), "polygon_points": 1024,
         "area_tolerance": 1e-5},
        {"criterion": "resolved_all_sampled_phases_above", "ratio": max_1024, "R5": 5 * max_1024,
         "phase_rad": max_phase, "phase_deg": np.degrees(max_phase), "polygon_points": 1024,
         "area_tolerance": 1e-5},
    ]
    for tolerance in (1e-6, 1e-4):
        values = np.array([_operational_threshold(p, area_tolerance=tolerance) for p in phases])
        for criterion, index, sign in (("resolved_no_sampled_phase_below", int(np.argmin(values)), 1),
                                       ("resolved_all_sampled_phases_above", int(np.argmax(values)), -1)):
            lower = max(-np.pi, phases[index] - step)
            upper = min(np.pi, phases[index] + step)
            result = minimize_scalar(
                lambda v: sign * _operational_threshold(v, area_tolerance=tolerance),
                bounds=(lower, upper), method="bounded", options={"xatol": 2e-4})
            p = float(result.x)
            ratio = _operational_threshold(p, polygon_points=1024, area_tolerance=tolerance)
            summary.append({"criterion": criterion, "ratio": ratio, "R5": 5 * ratio,
                            "phase_rad": p, "phase_deg": np.degrees(p),
                            "polygon_points": 1024, "area_tolerance": tolerance})
    write_csv(root / "fifth_harmonic_threshold_summary.csv", summary)


def harmonic_map(root):
    rows = []
    ratios = np.linspace(0, .30, 25)
    phases = np.linspace(-np.pi, np.pi, 17)
    th = -np.pi / 2 + (np.arange(512) + .371) * np.pi / 512
    for h in (5, 7):
        for phase in phases:
            for ratio in ratios:
                C = np.zeros(h + 1, complex)
                C[1], C[3], C[h] = 10, 1, ratio * np.exp(1j * phase)
                geom = crossings(spp_fourier(th, C))
                rows.append({"order": h, "amplitude_ratio_to_C3": ratio, "phase_rad": phase,
                             "polygon_points": 512, **geom})
    write_csv(Path(root) / "harmonic_geometry_map.csv", rows)
    write_transitions(rows, root)
    fifth_harmonic_thresholds(root)


def checks(root):
    """Independent numerical checks, including the dynamic-model tolerance audit."""
    th = np.linspace(-np.pi / 2, 3 * np.pi / 2, 4096, endpoint=False)
    wanted = np.zeros(10, complex)
    wanted[1], wanted[3], wanted[5] = 2 + .3j, .12 - .07j, -.02 + .04j
    s = sum(np.imag(wanted[n] * np.exp(1j * n * th)) for n in range(1, 10))
    np.testing.assert_allclose(coefficients(th, s, 9), wanted, atol=1e-14)
    z = spp_fourier(th, wanted)
    sp = sum(np.imag(1j * n * wanted[n] * np.exp(1j * n * th)) for n in range(1, 10))
    spp = sum(np.imag(-n * n * wanted[n] * np.exp(1j * n * th)) for n in range(1, 10))
    np.testing.assert_allclose(z, np.exp(-1j * th) * (sp - 1j * spp), atol=1e-12)

    t = (np.arange(1024) + .371) * 2 * np.pi / 1024
    assert crossings(np.exp(1j * t))["crossings"] == 0
    assert crossings(np.sin(t) + 1j * np.sin(2 * t))["crossings"] == 1

    elastic = evaluate(Config(model="elastic"))
    assert abs(elastic["summary"]["work_per_cycle_Pa"]) < 1e-11
    assert len(elastic["events"]) == 0
    geometry_comparison(elastic)
    assert elastic["summary"]["ODE_crossings_1024"] > 0

    verification = Config(name="recovery_verification", model="recovery", G=(0.5, 0.5),
                          yield_strain=(0.20, 0.30), recovery_time=1.0, breakdown=1.0,
                          ymin_fraction=0.35)
    a = evaluate(verification)
    b = evaluate(verification, npoints=8192, rtol=1e-10)
    err = float(np.max(np.abs(a["Z"] - b["Z"][::2])))
    assert err < 2e-5
    for d in (a, b, elastic, evaluate(Config())):
        assert d["summary"]["min_dissipation_Pa_per_s"] >= -1e-12
        assert d["summary"]["energy_identity_max_error"] < 1e-10
        assert abs(d["summary"]["work_per_cycle_Pa"]
                   - np.pi * d["config"].gamma0 ** 2 * d["C"][1].imag) < 1e-10
        assert np.min(d["lambda"]) >= -1e-8 and np.max(d["lambda"]) <= 1 + 1e-8
    geometry_comparison(a)
    geometry_comparison(b)
    assert a["summary"]["ODE_crossings_1024"] == b["summary"]["ODE_crossings_1024"]

    for r in (.121, .15, .20):
        angle = np.arccos(np.sqrt((25 * r - 3) / (40 * r)))
        cc = np.zeros(6, complex)
        cc[3], cc[5] = 1, r
        zz = spp_fourier(np.array([angle, np.pi - angle]), cc)
        np.testing.assert_allclose(zz[0], zz[1], atol=1e-12)

    critical_theta = -np.pi / 2 + (np.arange(2048) + .371) * np.pi / 2048
    for phase in np.linspace(-np.pi, np.pi, 25):
        below = np.zeros(6, complex)
        above = np.zeros(6, complex)
        below[3], below[5] = 1, .119 * np.exp(1j * phase)
        above[3], above[5] = 1, .201 * np.exp(1j * phase)
        assert crossings(spp_fourier(critical_theta, below), area_tolerance=0)["raw_crossings"] == 0
        assert crossings(spp_fourier(critical_theta, above), area_tolerance=0)["raw_crossings"] > 0

    report = {
        "status": "PASS",
        "checks": ["FFT phase convention", "SPP identity", "circle and figure-eight geometry",
                   "elastic false-positive control", "energy balance and nonnegative dissipation",
                   "work from first harmonic", "recovery bounds",
                   "ODE tolerance and sampling refinement", "analytic symmetric crossing",
                   "global fifth-harmonic threshold bounds"],
        "verification_model": asdict(verification),
        "refinement_Z_max_error_Pa": err,
        "verification_N_cross": a["summary"]["ODE_crossings_1024"],
        "elastic_crossings": elastic["summary"]["ODE_crossings_1024"],
    }
    (Path(root) / "checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "cases", "case", "sweep", "harmonics"))
    parser.add_argument("--config", help="JSON config file for the case command")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "runtime.json").write_text(json.dumps({
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform()}, indent=2), encoding="utf-8")
    if args.command == "case":
        if not args.config:
            parser.error("--config is required for the case command")
        single_case(args.out, args.config)
    else:
        {"check": checks, "cases": cases, "sweep": sweep, "harmonics": harmonic_map}[args.command](args.out)

