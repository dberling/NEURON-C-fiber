import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# tolerances
SPIKE_TIME_TOL = 0.001  # 1 ms
POTENTIAL_TOL = 0.01    # 1% relative difference

# --- Spike functions ---
def detect_spikes_from_csv(df):
    """
    Flatten all columns of spikes CSV into a 1D array for comparison.
    Assumes each column contains spike times.
    """
    return df.to_numpy().flatten()

def compare_spikes(spikes_new, spikes_ref, tol=SPIKE_TIME_TOL):
    """
    Compare spike arrays allowing a tolerance (ms).
    """
    if len(spikes_new) != len(spikes_ref):
        return False
    return np.all(np.abs(spikes_new - spikes_ref) <= tol)

# --- Potential functions ---
def compare_potential_trace(df_new, df_ref, tol=POTENTIAL_TOL):
    """
    Compare potential traces across all compartments.
    Time must match within relative tolerance.
    Voltage must match within relative tolerance.
    Assumes first column = 'Time', rest = voltage per compartment
    """
    # check time vector
    time_new = df_new.iloc[:, 0].values
    time_ref = df_ref.iloc[:, 0].values
    if not np.allclose(time_new, time_ref, rtol=tol, atol=0):
        return False

    # check voltage across all compartments (columns 1..n)
    volt_new = df_new.iloc[:, 1:].values
    volt_ref = df_ref.iloc[:, 1:].values
    if not np.allclose(volt_new, volt_ref, rtol=tol, atol=0):
        return False

    return True

# --- Main test ---
def test_run_simulation_qualitative(tmp_path):
    """
    Run the main simulation and verify that:
    - 3 CSV files are generated
    - potential traces match baseline within 1% (time + voltage)
    - spike times match baseline within 1 ms
    """

    project_root = Path(__file__).resolve().parents[1]

    # isolated Results directory
    results_dir = tmp_path / "Results"
    results_dir.mkdir()

    # run simulation
    completed = subprocess.run(
        [sys.executable, "run.py"],
        cwd=project_root,
        env={**dict(), "RESULTS_DIR": str(results_dir)},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"Simulation failed.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )

    # find CSV files
    csv_files = list(results_dir.glob("*.csv"))
    assert len(csv_files) == 3, f"Expected 3 CSV files, found {len(csv_files)}"

    # baseline folder
    baseline_dir = project_root / "tests" / "baseline_results"
    assert baseline_dir.exists(), "Baseline results folder missing!"

    for csv_file in csv_files:
        baseline_file = baseline_dir / csv_file.name
        assert baseline_file.exists(), f"Baseline for {csv_file.name} missing"

        df_new = pd.read_csv(csv_file)
        df_ref = pd.read_csv(baseline_file)

        if csv_file.name.startswith("spikes"):
            spikes_new = detect_spikes_from_csv(df_new)
            spikes_ref = detect_spikes_from_csv(df_ref)
            assert compare_spikes(spikes_new, spikes_ref), f"Spike times differ in {csv_file.name}"

        elif csv_file.name.startswith("potential"):
            assert compare_potential_trace(df_new, df_ref), f"Potential trace differs in {csv_file.name}"

        else:
            # ignore other CSVs
            pass

