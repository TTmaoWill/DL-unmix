"""Command-line interface; all outputs are written to new directories."""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from .api import DLUnmix, FitConfig, evaluate
from .synthetic import make_synthetic


def read_matrix(path, cts=False):
    # Preserve numeric-looking donor IDs and detect duplicate labels before pandas
    # can silently mangle them. Two headers explicitly encode gene/cell type.
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        first = next(reader, [])
        second = next(reader, []) if cts else None
        if len(first) < 2 or (cts and len(second) != len(first)):
            raise ValueError("invalid TSV header")
        labels = list(zip(first[1:], second[1:])) if cts else first[1:]
        components = first[1:] + (second[1:] if cts else [])
        if any(not label.strip() for label in components):
            raise ValueError("matrix column labels must be nonempty")
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate matrix column labels")
    f = pd.read_csv(path, sep="\t", header=[0, 1] if cts else 0, dtype=str, keep_default_na=False)
    ids = f.iloc[:, 0].tolist()
    f = f.iloc[:, 1:]
    f.index = pd.Index(ids, name="donor")
    if cts:
        f.columns = pd.MultiIndex.from_tuples(labels, names=["gene", "cell_type"])
        return {ct: f.xs(ct, axis=1, level=1).astype(float) for ct in dict.fromkeys(f.columns.get_level_values(1))}
    f.columns = labels
    return f.astype(float)


def write_cts(cts, path):
    f = pd.concat(cts, axis=1).swaplevel(0, 1, axis=1)
    f.columns.names = ["gene", "cell_type"]
    # A blank index name avoids a third header row in two-level TSV files.
    f.rename_axis(None).to_csv(path, sep="\t")


def _predict_outputs(model, bulk, fractions, out, floor, truth=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    p = model.predict(bulk, fractions, fraction_floor=floor)
    p.rename_axis(None).to_csv(out / "predictions.tsv", sep="\t")
    model.selected_profiles().to_csv(out / "selected_profiles.tsv", sep="\t", index_label="gene")
    (out / "prediction.json").write_text(json.dumps({"fraction_floor": floor, "expression_scale": "log2p1_nonnegative",
        "n_donors": len(p), "n_genes": len(model.genes_), "cell_types": model.cell_types_,
        "selected_profiles_threshold": 0.4, "predictions_filtered": False}, indent=2) + "\n")
    if truth is not None:
        evaluate(p, truth).to_csv(out / "evaluation.tsv", sep="\t", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dlunmix", description="Fit and apply the adopted reference-supervised DL-unmix model (CPU).")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    for name in ["reference-bulk", "reference-fractions", "reference-cts", "splits", "out"]:
        fit.add_argument("--" + name, required=True)
    fit.add_argument("--candidate-epochs", type=int, nargs="+", default=[5, 10, 15, 20])
    predict = commands.add_parser("predict")
    for name in ["model", "bulk", "fractions", "out"]:
        predict.add_argument("--" + name, required=True)
    predict.add_argument("--fraction-floor", type=float, default=0.01)
    predict.add_argument("--truth-cts", help="optional evaluation labels; never needed for prediction")
    demo = commands.add_parser("demo")
    demo.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            split = pd.read_csv(args.splits, sep="\t", dtype=str, keep_default_na=False)
            if set(split.columns) != {"donor", "split"} or not set(split["split"]).issubset({"train", "val", "refit_only"}):
                raise ValueError("splits TSV must have donor/split columns, with train, val or refit_only labels")
            model = DLUnmix(FitConfig(candidate_epochs=tuple(args.candidate_epochs))).fit(
                read_matrix(args.reference_bulk), read_matrix(args.reference_fractions), read_matrix(args.reference_cts, cts=True),
                train_donors=split.loc[split.split == "train", "donor"].tolist(),
                validation_donors=split.loc[split.split == "val", "donor"].tolist(),
                refit_only_donors=split.loc[split.split == "refit_only", "donor"].tolist())
            model.save(args.out)
            print(f"Saved model; selected {model.selected_epochs_} epochs")
        elif args.command == "predict":
            model = DLUnmix.load(args.model)
            truth = read_matrix(args.truth_cts, cts=True) if args.truth_cts else None
            _predict_outputs(model, read_matrix(args.bulk), read_matrix(args.fractions), args.out, args.fraction_floor, truth)
            print("Saved predictions and selection mask")
        else:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=False)
            reference, target = make_synthetic()
            for label, (bulk, fractions, truth) in [("reference", reference), ("target", target)]:
                bulk.to_csv(out / f"{label}_bulk.tsv", sep="\t", index_label="donor")
                fractions.to_csv(out / f"{label}_fractions.tsv", sep="\t", index_label="donor")
                write_cts(truth, out / f"{label}_cts.tsv")
            donors = reference[0].index.tolist()
            pd.DataFrame({"donor": donors, "split": ["train"]*12 + ["val"]*4}).to_csv(out / "splits.tsv", sep="\t", index=False)
            model = DLUnmix(FitConfig(candidate_epochs=(1, 2))).fit(*reference, train_donors=donors[:12], validation_donors=donors[12:])
            model.save(out / "model")
            _predict_outputs(DLUnmix.load(out / "model"), *target[:2], out / "prediction", 0.01, target[2])
            print("Synthetic software demonstration complete (candidate epochs 1,2; not a scientific benchmark)")
    except (ValueError, TypeError, FileExistsError, FileNotFoundError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
