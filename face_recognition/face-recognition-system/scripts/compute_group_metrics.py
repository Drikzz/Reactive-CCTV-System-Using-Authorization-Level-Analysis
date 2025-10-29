import argparse
import csv
from pathlib import Path
from collections import defaultdict, Counter
import math

def read_results(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({k: (v or "").strip() for k,v in r.items()})
    return rows

def read_gt(path):
    # Expect CSV with columns 'file' and 'ground_truth' (or filename,label)
    gt = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        rdr = csv.DictReader(f)
        cols = [c.lower() for c in rdr.fieldnames]
        file_col = None
        label_col = None
        for c in rdr.fieldnames:
            lc = c.lower()
            if lc in ("file","filename","image","img"): file_col = c
            if lc in ("ground_truth","ground truth","gt","label","identity","name"): label_col = c
        if file_col is None or label_col is None:
            raise SystemExit("GT CSV must contain a file and a ground_truth/label column")
        for r in rdr:
            key = Path(r[file_col]).name
            gt[key] = (r[label_col] or "").strip()
    return gt

def build_group_map_from_dataset(dataset_root):
    # Expect structure: dataset_root/<group>/<identity>/<images...>
    mapping = {}
    root = Path(dataset_root)
    if not root.exists():
        raise SystemExit("Dataset path not found: " + str(dataset_root))
    for group_dir in [d for d in root.iterdir() if d.is_dir()]:
        # if group_dir contains identity dirs
        for identity in [d for d in group_dir.iterdir() if d.is_dir()]:
            mapping[identity.name] = group_dir.name
    if not mapping:
        raise SystemExit("Could not infer group->identity mapping from dataset. Provide --group-map CSV or use grouped dataset layout.")
    return mapping

def read_group_map(path):
    # CSV with columns identity,group
    gm = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        rdr = csv.DictReader(f)
        fld = [c.lower() for c in rdr.fieldnames]
        id_col = None; g_col = None
        for c in rdr.fieldnames:
            lc = c.lower()
            if lc in ("identity","id","name","person"): id_col = c
            if lc in ("group","level","authorization","auth"): g_col = c
        if id_col is None or g_col is None:
            raise SystemExit("Group map CSV must contain identity and group columns")
        for r in rdr:
            gm[(r[id_col] or "").strip()] = (r[g_col] or "").strip()
    return gm

def safe_map_identity_to_group(identity, group_map):
    if not identity:
        return "Unknown"
    # exact match
    if identity in group_map:
        return group_map[identity]
    # try case-insensitive
    for k,v in group_map.items():
        if k.lower() == identity.lower():
            return v
    # fallback heuristics
    low = identity.lower()
    if "partial" in low:
        return "PartialAuthorized"
    if "auth" in low or "authorized" in low or "level" in low:
        return "Authorized"
    return "Unknown"

def metrics_from_confusion(conf, total):
    # conf: dict[group]->{'TP','FP','FN','TN'}
    res = {}
    for g,vals in conf.items():
        TP = vals['TP']; FP = vals['FP']; FN = vals['FN']; TN = vals['TN']
        precision = TP / (TP + FP) if (TP + FP) else float('nan')
        recall = TP / (TP + FN) if (TP + FN) else float('nan')
        f1 = (2*precision*recall/(precision+recall)) if (precision and recall and not math.isnan(precision) and not math.isnan(recall)) else float('nan')
        accuracy = (TP + TN) / total if total else float('nan')
        res[g] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "TP":TP,"FP":FP,"FN":FN,"TN":TN}
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="CSV from facenet_infer.py / facenet_infer_video.py (must contain 'file' or 'video'+'frame' and 'predicted')")
    p.add_argument("--gt", required=True, help="ground truth CSV mapping file -> identity (columns: file, ground_truth)")
    p.add_argument("--group-map", default="", help="optional CSV mapping identity->group (columns: identity, group)")
    p.add_argument("--dataset", default="", help="optional dataset root to infer groups: dataset/<group>/<identity>/<images>")
    args = p.parse_args()

    results = read_results(args.results)
    gt = read_gt(args.gt)

    # load or build group map
    if args.group_map:
        group_map = read_group_map(args.group_map)
    elif args.dataset:
        group_map = build_group_map_from_dataset(args.dataset)
    else:
        group_map = {}  # best-effort; identities not found -> Unknown

    # prepare pairs
    pairs = []
    notfound = 0
    for r in results:
        # try to find file key
        file_key = None
        if "file" in r and r["file"]:
            file_key = Path(r["file"]).name
        elif "filename" in r and r["filename"]:
            file_key = Path(r["filename"]).name
        elif "video" in r and "frame" in r:
            # create key like video__frame to match GT if user stored that way
            file_key = f"{Path(r['video']).name}__{r['frame']}"
        else:
            # try any first column that looks like a filename
            file_key = None
        if file_key is None:
            continue
        pred_identity = r.get("predicted","").strip()
        gt_identity = gt.get(file_key,"").strip()
        if not gt_identity:
            notfound += 1
            continue
        # map to groups
        pred_group = safe_map_identity_to_group(pred_identity, group_map)
        true_group = safe_map_identity_to_group(gt_identity, group_map) if group_map else (gt_identity or "Unknown")
        pairs.append((true_group, pred_group))

    if not pairs:
        print("No matching rows between results and ground-truth. Confirm file name keys.")
        return

    total = len(pairs)
    groups = sorted(set([a for a,b in pairs] + [b for a,b in pairs]))
    # initialize confusion stats
    conf = {g: {'TP':0,'FP':0,'FN':0,'TN':0} for g in groups}
    for true_g, pred_g in pairs:
        for g in groups:
            if true_g == g and pred_g == g:
                conf[g]['TP'] += 1
            elif true_g != g and pred_g == g:
                conf[g]['FP'] += 1
            elif true_g == g and pred_g != g:
                conf[g]['FN'] += 1
            else:
                conf[g]['TN'] += 1

    metrics = metrics_from_confusion(conf, total)

    # print table
    print("Per-group metrics (Precision / Recall / F1 / Accuracy) — total samples:", total)
    print("{:20s} {:9s} {:9s} {:9s} {:9s}".format("Group","Precision","Recall","F1","Accuracy"))
    for g in groups:
        m = metrics[g]
        prec = f"{m['precision']*100:.1f}%" if not math.isnan(m['precision']) else "n/a"
        rec = f"{m['recall']*100:.1f}%" if not math.isnan(m['recall']) else "n/a"
        f1s = f"{m['f1']:.3f}" if not math.isnan(m['f1']) else "n/a"
        acc = f"{m['accuracy']*100:.1f}%" if not math.isnan(m['accuracy']) else "n/a"
        print("{:20s} {:9s} {:9s} {:9s} {:9s}".format(g,prec,rec,f1s,acc))

    # save CSV
    out_csv = Path(args.results).with_suffix(".group_metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group","precision","recall","f1","accuracy","TP","FP","FN","TN"])
        for g in groups:
            m = metrics[g]
            w.writerow([g,
                        "" if math.isnan(m['precision']) else f"{m['precision']:.6f}",
                        "" if math.isnan(m['recall']) else f"{m['recall']:.6f}",
                        "" if math.isnan(m['f1']) else f"{m['f1']:.6f}",
                        "" if math.isnan(m['accuracy']) else f"{m['accuracy']:.6f}",
                        m['TP'], m['FP'], m['FN'], m['TN']])
    print("Saved per-group CSV to", out_csv)

if __name__ == "__main__":
    main()