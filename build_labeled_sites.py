## figure out consistent docstring style ###

import pysam
from pathlib import Path
import csv

from read_vcf_sites import read_vcf_sites
from get_ref_base import get_ref_base
from parse_one_site import parse_one_site


FIELDNAMES = ["chrom", "pos", "ref", "depth", "A", "C", "G", "T", "OTHER", "gt"]



def build_labeled_row(bam, fasta, site, require_ref_match=True):
    """
    Build one labeled training row from VCF site or None if unusable

    Args:
    bam: pysam.AlignmentFile, Input BAM file
    fasta: pyfasta.Fasta, Reference FASTA  
    site: dict, VCF site data
    require_ref_match: bool, Skip if FASTA ref != VCF ref

    Returns:
    dict or None, Training row dict
    """


    chrom =  site["chrom"]
    pos = site["pos"]

    #print("FASTA refs:", fasta.references) #debugging
    #print("VCF chrom:", chrom) #debugging
   
    #fasta ref at specific site
    ref = get_ref_base(fasta, chrom, pos)

    #skip ambiguous/ unusable reference bases
    if ref not in {"A", "C", "G", "T"}:
        return None

    #check against vcf reference allele
    vcf_ref = site.get("ref")
    if require_ref_match and vcf_ref is not None and ref != vcf_ref.upper():
        return None

    # get return list from  parse one site
    counts = parse_one_site(bam, chrom, pos) 

    # include OTHER in depths? (bc it is also observed base)
    depth = counts["A"] + counts["C"] + counts["G"] + counts["T"]

    # skip uncovered sites 
    if depth == 0:
        return None

    #return ONE dict line for training data
    return {
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "depth": depth,
        "A": counts["A"],
        "C": counts["C"],
        "G": counts["G"],
        "T": counts["T"],
        "OTHER": counts["OTHER"],
        "gt": site["gt"],
    }

############ GENERATOR FOR STREAMING #############
def labeled_row_generator(bam, fasta, vcf_path, require_ref_match=True):
    """
    Stream labeled rows one at a time from VCF sites

    Args:
    bam: Open BAM file handle
    fasta: Open FASTA file handle
    vcf_path: Path to VCF file
    require_ref_match: Skip FASTA/VCF ref mismatches (default: True)

    Yields:
    Row dicts for valid sites
    """

    for site in read_vcf_sites(vcf_path):
        row = build_labeled_row(
            bam=bam,
            fasta=fasta,
            site=site,
            require_ref_match=require_ref_match,
        )
        if row is not None:
            yield row

#write rows for csv that will be training rows 
def build_labeled_sites(bam_path, fasta_path, vcf_path, out_path, require_ref_match=True):
    """
    Build labeled training rows from VCF/BAM/FASTA, write to CSV

    Args:
    bam_path: Input BAM file path
    fasta_path: Reference FASTA path  
    vcf_path: Input VCF file path
    out_path: Output CSV path
    require_ref_match: Skip FASTA/VCF ref mismatches (default: True)

    Returns:
    Path to written CSV (max 1000 rows)
    """
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0  
    with pysam.AlignmentFile(bam_path, "rb") as bam, \
         pysam.FastaFile(fasta_path) as fasta, \
         open(out_path, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=FIELDNAMES) 
        writer.writeheader()

        max_rows = 1000
        
        for row in labeled_row_generator(
            bam=bam,
            fasta=fasta,
            vcf_path=vcf_path,
            require_ref_match=require_ref_match,
        ):
            if n_written >= max_rows:
                break
            
            writer.writerow(row)
            n_written += 1

    print(f"wrote {n_written} labeled rows to {out_path}")
    return out_path



######### TESTING #######
if __name__ == "__main__":
    BAM_PATH = "/projects/willerslev/scratch/bkl835/4Sabrina/Anzick1.bam"
    FASTA_PATH = "/projects/willerslev/scratch/bkl835/4Sabrina/hs.build37.1.fa"
    VCF_PATH = "/projects/willerslev/scratch/bkl835/4Sabrina/Anzick1_Imputed_from_msk.meanGP99flt.info05.chr10.vcf.gz"
    OUT_PATH = "outputs/chr10_labeled_sites.csv"

    build_labeled_sites(
        bam_path=BAM_PATH,
        fasta_path=FASTA_PATH,
        vcf_path=VCF_PATH,
        out_path=OUT_PATH,
        require_ref_match=True,
    )