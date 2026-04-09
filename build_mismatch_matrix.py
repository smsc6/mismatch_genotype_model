import csv
from pathlib import Path
# import argparse  --- REVISIT THIS LATER

# constants
BASES = ["A", "C", "G", "T"]
MISMATCH_COLS = [f"{r}_to_{b}" for r in BASES for b in BASES if r != b]


def build_mismatch_features(row):
    """
    Convert one labeled row to mismatch feature row
    """
    ref = row["ref"].upper()
    depth = int(row["depth"])

    counts = {b: int(row[b]) for b in BASES}

    mismatch_dict = {col: 0 for col in MISMATCH_COLS}

    n_match = counts[ref]

    for b in BASES:
        if b != ref:
            mismatch_dict[f"{ref}_to_{b}"] = counts[b]

    n_mismatch = sum(mismatch_dict.values())

    ### this causes runtime error if dividing by zero
    # match_rate = n_match / depth
    # mismatch_rate = n_mismatch / depth

    #replacement: 
    if depth == 0:
        match_rate = 0.0
        mismatch_rate = 0.0
    else:
        match_rate = n_match / depth
        mismatch_rate = n_mismatch / depth

    return {
        "chrom": row["chrom"],
        "pos": row["pos"],
        "ref": ref,
        "depth": depth,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "match_rate": match_rate,
        "mismatch_rate": mismatch_rate,
        **mismatch_dict,
        "gt": row["gt"],
    }


def mismatch_row_generator(in_csv):
    """
    Stream mismatch rows one at a time
    """
    with open(in_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in  reader:
            yield build_mismatch_features(row)



def build_mismatch_matrix(in_csv, out_csv):
    """
    Convert labeled_sites.csv to mismatch feature CSV
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_csv, "r") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)

        # build fieldnames dynamically from first row
        first = next(reader, None) # None for empty file error
        
        ##empty file edge case
        if first is None:
            print(f"No data rows in {in_csv}, writing empty output")

            fieldnames = ["chrom","pos","ref","depth","n_match","n_mismatch",
                          "match_rate","mismatch_rate", *MISMATCH_COLS, "gt",
            ]

            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            return out_path

        first_out = build_mismatch_features(first)
        fieldnames = [
            "chrom",
            "pos",
            "ref",
            "depth",
            "n_match",
            "n_mismatch",
            "match_rate",
            "mismatch_rate",
    *MISMATCH_COLS,
    "gt",
] 

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(first_out)

        n_written = 1

        for row in reader:
            out_row = build_mismatch_features(row)
            writer.writerow(out_row)
            n_written += 1

    print(f"Wrote {n_written} mismatch rows to {out_path}")
    return out_path



###########  TESTING ###########
if __name__ == "__main__":
    IN_CSV = "/projects/willerslev/people/vmk372/anzick_project/outputs/chr10_labeled_sites.csv"
    OUT_CSV = "/projects/willerslev/people/vmk372/anzick_project/outputs/chr10_mismatch_matrix.csv"

    build_mismatch_matrix(IN_CSV, OUT_CSV)