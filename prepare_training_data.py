import csv
from pathlib import Path


FEATURE_COLS = [
    "depth",
    "n_match",
    "n_mismatch",
    "match_rate",
    "mismatch_rate",
    "A_to_C", "A_to_G", "A_to_T",
    "C_to_A", "C_to_G", "C_to_T",
    "G_to_A", "G_to_C", "G_to_T",
    "T_to_A", "T_to_C", "T_to_G",
]

GT_TO_LABEL = { 
    "0/0": 0,
    "0/1": 1,
    "1/1": 2,
}


def prepare_training_row(row):
    """
    Convert one mismatch-feat row into model-ready training row

    Args:
    row: dict from mismatch matrix CSV

    Returns:
    Model-ready training row dict
    """

    gt = row["gt"]

    if gt not in GT_TO_LABEL:
        return None

    out_row = {}

    for col in FEATURE_COLS:
        
        #handle bad values better - can crash if empty/ non-num
        try:
            out_row[col] = float(row[col])
        except (KeyError, ValueError, TypeError):
            return None

    out_row["chrom"] = row["chrom"]
    out_row["pos"] = row["pos"]
    out_row["ref"] = row["ref"]
    out_row["gt"] = gt
    out_row["y"] = GT_TO_LABEL[gt]

    return out_row


def training_row_generator(in_csv):
    """
    Stream training rows one at a time

    Args:
    in_csv: Path to mismatch matrix CSV

    Yields:
    Model-ready row dicts
    """

    with open(in_csv, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            out_row = prepare_training_row(row)
            if out_row is not None:
                yield out_row


def prepare_training_data(in_csv, out_csv):
    """
    Convert mismatch feature CSV to model training CSV

    Args:
    in_csv: Path to mismatch matrix CSV
    out_csv: Path to output training CSV

    Returns:
    Output path
    """

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames =  ["chrom" , "pos", "ref"] + FEATURE_COLS + ["gt", "y"]
    n_written = 0
    

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in training_row_generator(in_csv):
            
            # if row is None:   (generator does this already)
            #     n_skipped += 1
            #     continue
            
            writer.writerow(row)
            n_written += 1

    print(f"Wrote {n_written} rows  to (path) {out_path}")
    return out_path


if __name__ == "__main__":
    IN_CSV = "/projects/willerslev/people/vmk372/anzick_project/outputs/chr10_mismatch_matrix.csv"
    OUT_CSV = "/projects/willerslev/people/vmk372/anzick_project/outputs/chr10_training_data_v1.csv"

    prepare_training_data(IN_CSV, OUT_CSV)