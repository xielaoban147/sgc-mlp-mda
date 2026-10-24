"""Prepare public archives from local submission artifacts, without altering originals.

Usage: python tools/refresh_artifacts.py --paper-dir PATH
The default output root is this repository. No network access or training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

DATE = "2026-10-24"
ARTIFACTS = (
    "ICAICE2026_reproduction_material.zip",
    "ICAICE2026_LaTeX_source.zip",
    "ICAICE2026_manuscript_draft.pdf",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, (2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, entries[name])


def public_reproduction(source, target):
    original = source.read_bytes()
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        entries = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    removed = sorted(n for n in entries if "__pycache__" in Path(n).parts or n.endswith((".pyc", ".pyo")))
    for name in removed:
        del entries[name]
    provenance = json.loads(entries["provenance.json"])
    old_copied = provenance.get("copied_files", [])
    new_copied = [r for r in old_copied if r.get("package_path") not in removed]
    if new_copied != old_copied:
        provenance["copied_files"] = new_copied
        entries["provenance.json"] = (json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode()
    note = {
        "date": DATE,
        "original_submission_archive_sha256": sha(original),
        "removed_entries": removed,
        "changes": ["Remove generated Python bytecode caches containing build-machine filenames.",
                    "Recompute the integrity manifest, covering every remaining file except itself.",
                    "Add this publication note. Data, experimental source code, logs and numeric results are byte-identical to the submission archive."],
        "source_archive_retained_unchanged": True,
    }
    entries["publication_note.json"] = (json.dumps(note, indent=2) + "\n").encode()
    manifest = json.loads(entries["manifest.json"])
    manifest["publication_date"] = DATE
    manifest["files_sha256"] = {n: sha(b) for n, b in sorted(entries.items()) if n != "manifest.json"}
    entries["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    archive(target, entries)
    original_entries = {}
    with zipfile.ZipFile(source) as z:
        original_entries = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    changed = sorted(n for n in entries if n in original_entries and entries[n] != original_entries[n])
    assert set(changed) <= {"manifest.json", "provenance.json"}
    for n, b in entries.items():
        if n.endswith(".py") or n.endswith((".npz", ".jsonl")):
            assert b == original_entries[n]
    return entries, {"source_sha256": sha(original), "public_sha256": sha(target.read_bytes()),
                     "removed_cache_count": len(removed), "modified_existing_files": changed,
                     "data_source_and_result_bytes_preserved": True}


PUBLIC_BUILD = r'''$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $latexmkCommand = (Get-Command latexmk -ErrorAction Stop).Source
    & $latexmkCommand -pdf -synctex=1 -interaction=nonstopmode -file-line-error -outdir=out main.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed: $LASTEXITCODE" }
    Copy-Item -LiteralPath 'out/main.pdf' -Destination 'ICAICE2026_manuscript_draft.pdf' -Force
} finally { Pop-Location }
'''

LATEX_README = """# Manuscript source

Working manuscript: *A Lightweight Interaction Decoder for Metabolite-Disease Ranking in Cold-Start Settings*.
Wei Xie; Wenzhi Liu (corresponding author), School of Mathematics and Computer Science,
Northwest Minzu University, Lanzhou, China. This is a research draft, not a claim of publication or acceptance.

The included PDF is four pages including references. Install a LaTeX distribution with
`latexmk`, `IEEEtran`, `gbt7714` numeric bibliography support, `amsmath`, `newtxmath` and
standard graphics packages. Compile with `latexmk -pdf -outdir=out main.tex` or, on Windows,
`./build.ps1` with `latexmk` on PATH. The PowerShell helper also copies `out/main.pdf`
to `ICAICE2026_manuscript_draft.pdf`.

The two vector figure PDFs are ready to compile; no Python or proprietary font installation
is needed to typeset them. Regenerating the figures requires NumPy, Matplotlib and a licensed
installation of Times New Roman. Figure scripts use this study's frozen scores and cite their
visual-design references in `figures/README.md`.

This public source archive excludes internal review correspondence and local QA reports.
Experimental data, exact splits, implementations and score evidence are in the separate
`ICAICE2026_reproduction_material.zip`. Upstream class/style copyright and license notices
are retained. No blanket license over third-party data or code is granted here.
"""


def public_latex(paper, target):
    names = ["main.tex", "references.bib", "IEEEtran.cls", "IEEEtran.bst",
             "sections/data_task.tex", "ICAICE2026_manuscript_draft.pdf"]
    names += ["figures/" + n for n in ["main_ap.pdf", "decoder_ablation.pdf", "main_ap.png",
        "decoder_ablation.png", "experiment_plot_data.json", "ablation_plot_data.json",
        "make_experiment_figures.py", "make_main_ap_figure.py", "README.md"]]
    entries = {n: (paper / n).read_bytes() for n in names}
    entries["README.md"] = LATEX_README.encode()
    entries["build.ps1"] = PUBLIC_BUILD.encode()
    entries[".latexmkrc"] = b"# Use the TeX tools available on PATH.\n$pdf_mode = 1;\n"
    archive(target, entries)
    return entries


def scan(entries):
    findings = {}
    patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "user_home_path": r"[A-Za-z]:[\\/]Users[\\/][^\r\n\"']+",
        "credential": r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)",
    }
    for category, pattern in patterns.items():
        found = []
        for n, b in entries.items():
            if Path(n).suffix in {".py", ".md", ".json", ".jsonl", ".txt", ".log", ".tex", ".bib", ".ps1"}:
                if re.search(pattern, b.decode("utf-8", "replace")):
                    found.append(n)
        findings[category] = found
    assert not any(findings.values()), findings
    assert not any(Path(n).is_absolute() or ".." in Path(n).parts for n in entries)
    return findings


def refresh(paper, output):
    output.mkdir(parents=True, exist_ok=True)
    downloads = output / "downloads"
    downloads.mkdir(exist_ok=True)
    repro, audit = public_reproduction(paper / ARTIFACTS[0], downloads / ARTIFACTS[0])
    latex = public_latex(paper, downloads / ARTIFACTS[1])
    shutil.copyfile(paper / ARTIFACTS[2], downloads / ARTIFACTS[2])
    for n, b in repro.items():
        if n.endswith(".py") or n in {"protocol.md", "experiments/ctd_sgformer/requirements.lock.txt", "experiments/ctd_sgformer/adaptation_notes.md"}:
            destination = output / "source" / n
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b)
    checksums = {"downloads/" + n: sha((downloads / n).read_bytes()) for n in ARTIFACTS}
    (output / "checksums.sha256").write_text("".join(f"{v}  {n}\n" for n, v in checksums.items()), encoding="ascii")
    manifest = {"date": DATE, "repository": "https://github.com/xielaoban147/sgc-mlp-mda",
        "artifact_sha256": checksums, "public_reproduction_audit": audit,
        "reproduction_content_scan": scan(repro), "latex_content_scan": scan(latex),
        "source_file_count": sum(1 for n in repro if n.endswith(".py")),
        "submission_originals_modified": False,
        "distribution": "The three files under downloads/ are intended as GitHub Release assets; source/ is a browseable mirror, not a separate dataset distribution.",
        "license": "No blanket third-party code or database license is granted. Retained upstream notices and source-database terms apply."}
    verification_path = output / "validation/saved_score_verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        manifest["saved_score_verification"] = {k: v for k, v in verification.items() if k != "details"}
    write_json(output / "publication_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(refresh(args.paper_dir.resolve(), args.payload_dir.resolve()), indent=2))
