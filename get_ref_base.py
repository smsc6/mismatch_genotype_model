##### RETURN AND VERIFY/ CHECK EDGE CASES? ######

import pysam


def get_ref_base(fasta, chrom, pos):
    """
    Get reference base at 1-based genomic position

    Args:
    fasta: Open FASTA handle
    chrom: Chromosome name  
    pos: 1-based position

    Returns:
    Reference base (A/C/G/T/N)
    """
    
    #shift position 1 because vcf/bam 1-based, fasta 0-based (off by 1 err)
    return fasta.fetch(chrom, pos - 1, pos).upper() #takes already open fasta handle


############# TESTING- test @ chrom 10/ 11 ###################### 
if __name__ == "__main__":
    FASTA_FILE = "PATH_TO_FASTA"
    CHROM = "1"
    POS = 10000673

    with pysam.FastaFile(FASTA_FILE) as fasta:
        print(get_ref_base(fasta, CHROM, POS))