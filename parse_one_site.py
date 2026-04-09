from collections import Counter

def parse_one_site(bam, chrom, pos):
    """
    :param:
    :return: 
    """
    counts = Counter({"A": 0, "C": 0, "G": 0, "T": 0, "OTHER": 0})

    for col in bam.pileup(chrom, pos - 1, pos, truncate=True, stepper="all"):
        if col.reference_pos != pos - 1:
            continue

        for pr in col.pileups:
            if pr.is_del or pr.is_refskip or pr.query_position is None:
                continue

            seq = pr.alignment.query_sequence
            if seq is None:
                continue

            qpos = pr.query_position
            if qpos < 0 or qpos >= len(seq):
                continue

            base = seq[qpos].upper()
            counts[base if base in "ACGT" else "OTHER"] += 1

        return counts

    return counts




############# TESTING  ##########
if __name__ == "__main__":
    import pysam

    BAM_PATH = "BAM_PATH"
    CHROM = "10"
    POS = 30015672

    with pysam.AlignmentFile(BAM_PATH, "rb") as bam:
        counts = parse_one_site(bam, CHROM, POS)

    depth = counts["A"] + counts["C"] + counts["G"] + counts["T"]

    print(f"Chrom {CHROM}: Pos {POS}")
    print(f"Depth: {depth}")
    print("Counts:", {b: counts[b] for b in ["A", "C", "G", "T"]})
    print("Other:", counts["OTHER"])