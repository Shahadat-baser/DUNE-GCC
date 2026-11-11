"""
DUNE v3 - automatic package check and install.  Every step calls  dune_env.ensure("<step>")  BEFORE it
imports numpy / pandas / geopandas / rasterio, so a fresh environment is set up by simply running the steps.

What it does for each package the step needs:
  1. is it installed in THIS kernel's Python, at an allowed version?           -> nothing to do
  2. if not, pip-installs the pinned version into this Python (sys.executable)
  3. test-imports it in a separate process, so a DLL blocked by Windows Application Control is
     detected without crashing the notebook; if blocked, it tries the next older pinned version
  4. if a package that is ALREADY loaded in this kernel was changed, it stops and asks for a kernel
     restart (Python cannot swap a loaded compiled library)

Pins: the first version in each list is the one known to load on a Windows PC with Application Control (pandas 3.x was blocked).
Only the standard library is used here.
"""
import os, sys, subprocess, importlib, importlib.metadata as md
import dune_config as C

# distribution -> (import name, allowed check, versions to try in order)
PKGS = {
    "earthengine-api": ("ee",         lambda v: _v(v) >= (1, 4),             ["1.7.46", "1.7.40"]),
    "requests":        ("requests",   lambda v: _v(v) >= (2, 31),            ["2.32.5", "2.32.3"]),
    "truststore":      ("truststore", lambda v: True,                         ["0.10.4"]),
    "numpy":           ("numpy",      lambda v: (1, 26) <= _v(v) < (2, 3),    ["2.2.6", "1.26.4"]),
    "pandas":          ("pandas",     lambda v: (2, 1) <= _v(v) < (3, 0),     ["2.3.3", "2.2.3"]),
    "shapely":         ("shapely",    lambda v: _v(v) >= (2, 0),             ["2.1.2", "2.0.7"]),
    "pyogrio":         ("pyogrio",    lambda v: _v(v) >= (0, 7),             ["0.11.1", "0.10.0"]),
    "pyproj":          ("pyproj",     lambda v: _v(v) >= (3, 6),             ["3.7.2", "3.7.1"]),
    "geopandas":       ("geopandas",  lambda v: _v(v) >= (0, 14),            ["1.1.1", "1.0.1"]),
    "rasterio":        ("rasterio",   lambda v: _v(v) >= (1, 3),             ["1.4.4", "1.4.3"]),
    "scipy":           ("scipy",      lambda v: _v(v) >= (1, 11),            ["1.15.3", "1.14.1"]),
    "xgboost":         ("xgboost",    lambda v: (2, 0) <= _v(v) < (3, 0),    ["2.1.4", "2.0.3"]),
    "scikit-learn":    ("sklearn",    lambda v: _v(v) >= (1, 3),             ["1.6.1", "1.5.2"]),
    "matplotlib":      ("matplotlib", lambda v: _v(v) >= (3, 7),             ["3.10.0", "3.9.4"]),
    "python-docx":     ("docx",       lambda v: _v(v) >= (1, 0),             ["1.1.2", "1.1.0"]),
    "torch":           ("torch",      lambda v: (2, 3) <= _v(v) < (3, 0),    ["2.5.1", "2.6.0", "2.4.1"]),   # CPU wheel from PyPI
}
BASE = ["earthengine-api", "requests", "numpy", "pandas"] + (["truststore"] if C.SSL_MODE == "truststore" else [])
GEO = ["shapely", "pyproj", "pyogrio", "geopandas"]
STEPS = {
    "preflight":        BASE,
    "upload_districts": BASE + GEO,
    "build_inputs":     BASE + GEO,
    "static":           BASE + GEO + ["rasterio", "scipy"],
    "station_daily":    BASE,
    "district_daily":   BASE,
    "weekly_rasters":   BASE + ["rasterio"],
    "ground_truth":     BASE + GEO,
    "ground_download":  BASE,
    "features":         BASE + ["rasterio"],
    "xgboost":          BASE + ["scipy", "xgboost"],
    "maps":             BASE + GEO + ["rasterio", "scipy", "xgboost"],
    "lstm":             BASE + GEO + ["rasterio", "scipy", "xgboost", "torch"],
    "figures":          BASE + GEO + ["rasterio", "scipy", "xgboost", "matplotlib"],
    "tables":           BASE + GEO + ["rasterio", "scipy", "xgboost", "python-docx"],
    "diagnostics":      BASE + GEO + ["rasterio", "scipy", "xgboost", "scikit-learn", "matplotlib"],
}


def _v(s):
    out = []
    for p in str(s).split(".")[:3]:
        num = "".join(ch for ch in p if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out + [0] * (3 - len(out)))


def _installed(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _imports_ok(mod):
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1]


CONSTRAINTS = os.path.join(C.BASE_DIR, "logs", "pip_constraints.txt")


def _pip(spec):
    os.makedirs(os.path.dirname(CONSTRAINTS), exist_ok=True)
    with open(CONSTRAINTS, "w") as fh:                # never let a dependency pull pandas 3 / numpy 2.3+
        fh.write("pandas<3\nnumpy<2.3\n")
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-c", CONSTRAINTS, spec]
    if C.SSL_MODE == "off":                       # university HTTPS inspection
        for h in ["pypi.org", "files.pythonhosted.org", "pypi.python.org"]:
            cmd += ["--trusted-host", h]
    print(f"   pip install {spec}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("   " + "\n   ".join(r.stderr.strip().splitlines()[-4:]))
    return r.returncode == 0


def ensure(step):
    need = STEPS[step]
    print(f"[env] checking {len(need)} packages for step '{step}' in {sys.executable}")
    changed_loaded = []
    for dist in need:
        mod, ok_version, pins = PKGS[dist]
        have = _installed(dist)
        if have and ok_version(have):
            good, err = _imports_ok(mod)
            if good:
                continue
            print(f"   {dist} {have} is installed but cannot be imported: {err}")
        else:
            print(f"   {dist}: {'not installed' if not have else have + ' (version not allowed)'}")
        fixed = False
        for pin in pins:
            if not _pip(f"{dist}=={pin}"):
                continue
            good, err = _imports_ok(mod)
            if good:
                print(f"   {dist} {pin} OK")
                fixed = True
                break
            print(f"   {dist} {pin} installed but import fails: {err}")
        if not fixed:
            raise SystemExit(
                f"\n[env] {dist} could not be installed in a form this PC allows to load.\n"
                f"      Last error: {err if have else 'pip failed'}\n"
                f"      If it says 'Application Control policy', your IT administrator must allow Python packages in\n"
                f"      {os.path.dirname(sys.executable)}  - see the message above.")
        if mod in sys.modules:
            changed_loaded.append(dist)
    importlib.invalidate_caches()
    if changed_loaded:
        raise SystemExit(f"\n[env] {', '.join(changed_loaded)} changed while already loaded in this kernel.\n"
                         f"      Restart the kernel (VS Code: Restart) and run this cell again.")
    print("[env] all packages ready")
