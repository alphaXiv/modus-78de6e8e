# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.15",
#   "matplotlib==3.10.3",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # MODUS released-checkpoint reproduction

    MODUS proposes one decoder that translates among text, images, geometry,
    edges, and grounding. We attempted to audit the shared route, compare
    direct and Canny-chained surface normals on NYUv2, and test best-of-four
    self-verification on GenEval.

    **Verdict: blocked before model construction.** Static code evidence
    aligned with the shared-routing design, but no behavioral output was
    produced. This notebook embeds the evidence so opening it does not rerun
    expensive experiments.
    """)
    return


@app.cell
def _():
    claims = [
        {
            "claim": "Shared decoder + registry",
            "paper": "One released decoder; 16 modalities",
            "observed": "1 inferencer class; 16 registry entries",
            "evidence_level": 0.5,
            "assessment": "Static aligned; runtime blocked",
        },
        {
            "claim": "Canny chain on NYUv2",
            "paper": "20.02° direct vs 19.87° chained",
            "observed": "0/16 images reached inference",
            "evidence_level": 0.0,
            "assessment": "Inconclusive",
        },
        {
            "claim": "Self-verification on GenEval",
            "paper": "0.81 base, 0.82 grounding, 0.84 combined",
            "observed": "0/8 prompts reached generation",
            "evidence_level": 0.0,
            "assessment": "Inconclusive",
        },
    ]
    return (claims,)


@app.cell
def _(claims):
    import matplotlib.pyplot as plt

    labels = [row["claim"] for row in claims]
    levels = [row["evidence_level"] for row in claims]
    colors = ["#e5a93d" if value else "#9aa3ad" for value in levels]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.barh(labels, [1] * len(labels), color="#eef1f4", height=0.58)
    ax.barh(labels, levels, color=colors, height=0.58)
    ax.set_xlim(0, 1)
    ax.set_xticks(
        [0, 0.5, 1],
        ["No behavioral evidence", "Static code evidence", "Behavior measured"],
    )
    ax.invert_yaxis()
    ax.set_title("Evidence stopped before released-model behavior")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#d8dde3")
    fig.tight_layout()
    fig
    return


@app.cell
def _(claims, mo):
    mo.ui.table(claims, selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Where execution stopped

    The Kubernetes jobs allocated real RTX PRO 6000 Blackwell GPUs and
    downloaded the exact public checkpoint. The minimized environment exposed
    missing Python dependencies; the full released requirements then exposed
    missing native OpenCV libraries. After `libGL.so.1` was repaired, the
    second allowed attempt stopped on `libgthread-2.0.so.0`, before model
    construction.
    """)
    return


@app.cell
def _():
    runtime_stages = [
        ("Repository route audit", "passed"),
        ("Kubernetes GPU allocation", "passed"),
        ("Public checkpoint download", "passed"),
        ("Full Python requirements", "passed"),
        ("Checkpoint construction", "blocked"),
        ("Task inference", "not reached"),
        ("Metrics", "not reached"),
    ]
    return (runtime_stages,)


@app.cell
def _(mo, runtime_stages):
    rows = [{"stage": stage, "status": status} for stage, status in runtime_stages]
    mo.ui.table(rows, selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Exact planned tests

    - **NYUv2:** 16 evenly spaced examples from the public 654-image validation
      split; seeds `20260729 + index`; 512-pixel output; paired angular MAE,
      latency, and peak memory at 5, 10, and 20 denoising steps.
    - **GenEval:** eight fixed object-only prompts; four candidates per prompt;
      MODUS grounding-coordinate likelihood for selection; fixed random,
      edge-density, and oracle comparisons; COCO-DETR used only after selection.

    These protocols are included to make the boundary reproducible. They are
    not reported as model evidence because zero examples completed.
    """)
    return


@app.cell
def _():
    compute = {
        "backend": "Kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "peak_concurrent_gpus": 16,
        "elapsed_wall_hours": 0.82,
        "successful_behavioral_gpu_runs": 0,
        "successful_cpu_static_audits": 1,
    }
    return (compute,)


@app.cell
def _(compute, mo):
    mo.md(
        f"""
        ## Compute record

        - Backend: **{compute["backend"]}**
        - GPU model: **{compute["gpu_model"]}**
        - Peak concurrent allocation: **{compute["peak_concurrent_gpus"]} GPUs**
        - Actual elapsed wall time: **{compute["elapsed_wall_hours"]:.2f} hours**
        - Successful behavioral GPU runs: **{compute["successful_behavioral_gpu_runs"]}**

        The successful terminal evidence was CPU-only because static AST/YAML
        inspection does not benefit from a GPU.
        """
    )
    return


if __name__ == "__main__":
    app.run()
