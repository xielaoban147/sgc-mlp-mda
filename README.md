# SGC-MLP for metabolite–disease association ranking

Research materials for **A Lightweight Interaction Decoder for Metabolite-Disease Ranking in Cold-Start Settings**.

**Wei Xie**, **Wenzhi Liu** (corresponding author)  
School of Mathematics and Computer Science, Northwest Minzu University, Lanzhou, China.

This repository accompanies a research manuscript. The manuscript is a draft; no publication, acceptance, DOI or independent external validation is claimed.

## Downloads

The complete files are distributed as [GitHub Release assets](https://github.com/xielaoban147/sgc-mlp-mda/releases/latest).

| File | Contents |
|---|---|
| [Reproducibility archive](https://github.com/xielaoban147/sgc-mlp-mda/releases/latest/download/ICAICE2026_reproduction_material.zip) | Derived cohort, ten frozen splits, complete model and evaluation code, dependency lock, configuration/epoch records, seed-level scores and decoder-feature controls |
| [LaTeX source](https://github.com/xielaoban147/sgc-mlp-mda/releases/latest/download/ICAICE2026_LaTeX_source.zip) | Manuscript source, references, two vector experimental figures, plot data and build instructions |
| [Four-page manuscript PDF](https://github.com/xielaoban147/sgc-mlp-mda/releases/latest/download/ICAICE2026_manuscript_draft.pdf) | Current manuscript, including references |

SHA-256 checksums are in [`checksums.sha256`](checksums.sha256); packaging and integrity information is in [`publication_manifest.json`](publication_manifest.json). The public reproducibility archive removes a generated Python bytecode cache from the submission package and updates its integrity manifest. Experimental source, data and result bytes are preserved. Its `publication_note.json` records the original archive checksum.

[`source/`](source/) is a browseable mirror of the archived implementation and protocol. **Extract the reproducibility archive to run the study**: the mirror alone omits the numeric dataset, split arrays and score evidence.

## Method and task

SGC-MLP uses external chemical and disease descriptors, one-hop simplified graph convolution, and a 256–64–1 pairwise decoder. The decoder concatenates 64-dimensional metabolite and disease embeddings, their elementwise product, and their absolute difference. It has **484,033 allocated trainable parameters**, versus 467,520 for SGC-Dot and 497,088 for the SGFormer adaptations.

The structure-checked CTD–ChEBI cohort contains **234 metabolites, 433 diseases and 3,184 curated positive associations**. The remaining **98,138 pairs are unannotated**, rather than experimentally confirmed negatives. Chemical structures provide 2,048-dimensional Morgan features; independent MEDIC ancestors and disease names provide 1,536-dimensional hashed disease features.

Two protocols each use seeds **17, 42, 2026, 31415 and 27182**:

- **Random-edge holdout:** 2,547/318/319 positive pairs in training/validation/test; 47,795/47,796 unannotated validation/test pairs. These candidate sets are subsets of the Cartesian product.
- **Disease-ID cold start:** disjoint rosters of 303/64/66 training/validation/test diseases. Every validation/test disease ranks all 234 metabolites. All node identities and independent features are available during fitting; held-out diseases have no association-training supervision or context edges. This is **transductive, feature-known, association-unseen** ranking, rather than testing nodes first introduced after training.

Training fixes one unannotated pair per positive, sampled without replacement from the complement of all recorded positives. Cold training uses training diseases only. Positive minibatch queries are removed in both graph directions. Validation/test use the complete training graph without dropout or additional context deletion. Checkpoints and learning rates are chosen on validation disease-macro AP. Exact split counts, SGF-B context budgets, minibatching and selection rules are in [`source/protocol.md`](source/protocol.md).

## Results

Values are mean ± sample standard deviation over five repeated holdouts; AP is disease-macro average precision. These seeds are repeated holdouts, not cross-validation folds.

| Main method | Random-edge AP | Cold-disease AP |
|---|---:|---:|
| SGC-Dot | 0.2933 ± 0.0144 | 0.2804 ± 0.0216 |
| SGF-T | 0.2884 ± 0.0173 | 0.2875 ± 0.0174 |
| SGF-B | **0.2965 ± 0.0187** | 0.2805 ± 0.0206 |
| SGC-MLP | 0.2939 ± 0.0139 | **0.2962 ± 0.0314** |

SGF-T selects its local fusion weight on validation data; SGF-B uses a fixed weight and disease-centered training context budgets. SGC-MLP has the highest mean cold-disease AP among these four main methods, while SGF-B leads random-edge AP. SGC-MLP improves cold-disease AP over SGC-Dot on all five seeds. No claim of leading every protocol or metric is made.

The complete decoder-feature ablation is reported below. **E** denotes ordered endpoint embeddings, **P** products and **D** absolute differences. Removed blocks are zeroed; allocated decoder dimensions are unchanged, with zero-input columns inactive.

| Protocol | Inputs | AP | Recall@20 | AUROC |
|---|---|---:|---:|---:|
| Random edge | E | 0.2915 ± 0.0143 | 0.7189 ± 0.0152 | 0.8453 ± 0.0078 |
| Random edge | E+P | **0.2952 ± 0.0175** | 0.7126 ± 0.0156 | 0.8464 ± 0.0075 |
| Random edge | E+D | 0.2922 ± 0.0154 | 0.7281 ± 0.0217 | 0.8481 ± 0.0101 |
| Random edge | E+P+D | 0.2939 ± 0.0139 | **0.7311 ± 0.0223** | **0.8525 ± 0.0105** |
| Cold disease | E | 0.2841 ± 0.0197 | 0.5564 ± 0.0199 | 0.8578 ± 0.0095 |
| Cold disease | E+P | 0.2960 ± 0.0276 | 0.5581 ± 0.0170 | **0.8606 ± 0.0157** |
| Cold disease | E+D | 0.2832 ± 0.0205 | 0.5582 ± 0.0169 | 0.8544 ± 0.0109 |
| Cold disease | E+P+D | **0.2962 ± 0.0314** | **0.5714 ± 0.0158** | 0.8583 ± 0.0101 |

Products provide most of the cold-disease AP gain over endpoint-only scoring. Adding absolute differences to E+P changes mean cold AP by approximately 0.0002 and improves 2/5 seeds. E+P has higher mean random-edge AP than the complete scorer. All three predefined reduced variants are retained. Earlier results from these holdouts informed method development; the component study uses the existing frozen splits rather than an independent external cohort.

## Install and verify saved results

Download and extract the reproducibility archive into a fresh directory. Use Python 3.12; the original environment used 3.12.14. For a lightweight **score-only audit**, install NumPy and scikit-learn, then run:

```bash
python -m pip install numpy scikit-learn
python -B verify.py
```

`verify.py` checks the manifest, recomputes metrics for **70 saved test-score groups**, and verifies all eight feature-ablation summary groups. It does not train a model, import Torch or require a GPU. [`validation/saved_score_verification.json`](validation/saved_score_verification.json) records the public package's verification.

For full reruns, install the locked environment from the extracted root. The original lock includes `torch==2.8.0+cu126`:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -r experiments/ctd_sgformer/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
python reproduce.py --phase smoke --device cpu
```

For CPU-only training, substitute the official CPU build of Torch 2.8.0 for the CUDA build and retain the other locked versions. Different devices or library builds can change floating-point training trajectories.

Run the following phases as **separate processes, in this order**. The smoke phase is a two-epoch engineering check and contributes no reported result. Add `--device cpu` to each command if desired; the default uses one available GPU, otherwise CPU.

```bash
python reproduce.py --phase baselines-fit
python reproduce.py --phase baselines-test
python reproduce.py --phase full-fit
python reproduce.py --phase full-test
python reproduce.py --phase ablation-prepare
python reproduce.py --phase ablation-train
python reproduce.py --phase ablation-test
```

The baseline grid has 100 fits, the complete decoder 20 fits, and three reduced scorers 60 fits. Jobs run serially and write to fresh `rerun/` directories. Selection is frozen before test evaluation; completed test evaluation blocks retuning of that phase. The original archived evidence remains unchanged. See the archive's README for detailed orchestration and score-reconstruction provenance.

## Attribution and use terms

Data derive from [CTD](https://ctdbase.org/), [ChEBI](https://www.ebi.ac.uk/chebi/) and their referenced chemical/disease identity sources. Raw API caches are excluded. Source-database attribution and use terms apply; this project does not claim ownership of database records or grant a new license over them.

The SGFormer adaptation retains the upstream MIT notice in `models.py`; its original source is [qitianwu/SGFormer](https://github.com/qitianwu/SGFormer), pinned commit `3578e101c701491ce068bf26a9b029d2134903be`. [Adaptation notes](source/experiments/ctd_sgformer/adaptation_notes.md) identify the changes. PyTorch Geometric is a dependency rather than a vendored repository. No blanket license over third-party data, IEEE template files or upstream code is asserted. Authors have not selected an additional license for original project material.

Citation metadata are in [`CITATION.cff`](CITATION.cff). No DOI or conference acceptance is asserted.
