"""CPU-only integrity and saved-score verification; no checkpoint needed."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
HOME=Path(__file__).resolve().parent
EXP=HOME/"experiments/ctd_sgformer"
sys.path.insert(0,str(EXP))
from protocol import disease_metrics
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(scores,expected,split):
    with np.load(scores,allow_pickle=False) as s, np.load(EXP/"revision_v2/splits"/f"{split}.npz",allow_pickle=False) as d:
        assert np.array_equal(s["pairs"],d["test_pairs"]),str(scores)
        assert np.array_equal(s["labels"],d["test_labels"]),str(scores)
        actual=disease_metrics(s["pairs"],s["labels"],s["logits"],int(d["num_d"]))
    errors={k:float(actual[k]-expected[k]) for k in ("macro_ap","macro_recall20","macro_auroc")}
    tolerance=2e-6 if "audit_predictions" in str(scores) else 1e-12
    if any(abs(x)>tolerance for x in errors.values()):raise AssertionError((str(scores),errors))
    return {"split":split,"scores":scores.relative_to(HOME).as_posix(),"metric_differences":errors,"tolerance":tolerance}
def verify():
    manifest=read(HOME/"manifest.json")
    for relative,digest in manifest["files_sha256"].items():
        if sha(HOME/relative)!=digest:raise AssertionError("SHA mismatch: "+relative)
    checks=[]
    baseline=HOME/"evidence/baselines"
    for row in read(baseline/"all_results.json"):
        if row["method"] not in ("sgc","fixed_tuned","fixed_budget"):continue
        scores=baseline/"fits"/row["split"]/row["method"]/f"config{row['config_index']}"/"test_predictions.npz"
        checks.append(compare(scores,row["test"],row["split"]))
    for row in read(HOME/"evidence/sgc_mlp/test_results.json"):
        checks.append(compare(HOME/"evidence/sgc_mlp/audit_predictions"/f"{row['split']}.npz",row["test"],row["split"]))
    review=EXP/"reviewer_revision_20261023"
    for row in read(review/"test_results_all_variants.json"):
        if row["variant"]=="full":continue
        scores=review/"test_once"/f"{row['split']}_{row['variant']}_predictions.npz"
        checks.append(compare(scores,row["test"],row["split"]))
    assert len(checks)==70
    summary=read(review/"ablation_summary.json")
    rows=read(review/"test_results_all_variants.json")
    for group in summary:
        matching=[r for r in rows if r["variant"]==group["variant"] and r["split"].startswith(group["setting"])]
        assert len(matching)==5
        for metric in ("macro_ap","macro_recall20","macro_auroc"):
            values=np.array([r["test"][metric] for r in matching])
            assert abs(float(values.mean())-group["metrics"][metric]["mean"])<1e-12
            assert abs(float(values.std(ddof=1))-group["metrics"][metric]["sample_std"])<1e-12
    return {"all_manifest_sha256_match":True,"file_count":len(manifest["files_sha256"]),
            "test_score_groups_recomputed":len(checks),"new_ablation_summary_groups_verified":len(summary),
            "maximum_absolute_metric_difference":max(abs(v) for x in checks for v in x["metric_differences"].values()),"details":checks}
if __name__=="__main__":print(json.dumps(verify(),ensure_ascii=False,indent=2))
