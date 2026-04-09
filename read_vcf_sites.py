###THIS IS ONLY BIALLELLIC - NEED OT RETURN AND MAKE MULTI-ALLELIC AFTER BUILDING MODEL #####
### DON'T LOVE THIS; RETURN LATER AND ADD DEPTH/ FILTER###


import pysam

def read_vcf_sites(vcf_file):
    """
    :param: vcf_file: path to vcf file
    :yield: iterates through ..
    """

    with pysam.VariantFile(vcf_file) as vcf:
        
        #safety check for missing samples
        if len(vcf.header.samples) == 0:
            raise ValueError("VCF has no samples")
        
        sample = list(vcf.header.samples)[0]

        #loop over records
        for record in vcf:
            
            #skip non biallelic/ outlier records first
            if len(record.alts or []) != 1: #need this for making clean to begin w, but does it void data validity????
                continue

            gt = record.samples[sample]["GT"]

            #maybe not necessary
            # if len(record.samples) == 0:
            #     continue

            #skip missing, GT = None or multi-allelic for now
            if gt is None or None in gt:
                continue
            if any(allele not in (0,1) for allele in gt):
                continue

            #convert (0,0) to 0/0 - tuple to string, for use downstrem
            if gt == (0, 0):
                gt_label = "0/0"
            elif gt in [(0, 1), (1, 0)]:
                gt_label = "0/1"
            elif gt == (1, 1):
                gt_label = "1/1"
            else:
                continue

            # MAKE GENERATOR INSTEAD OF USING MEMORY
            yield {"chrom": record.chrom, "pos": record.pos, "ref": record.ref, "alt": record.alts[0], "gt": gt_label} #needs space after yield

        #vcf.close() not necessary - 'with' closes it


###########  TESTING ##################
if __name__ == "__main__":
    
    VCF_FILE = "PATH_TO_VCF"

    for i, site in enumerate(read_vcf_sites(VCF_FILE)):
        if i >= 1000:
            break
        print(f"{site['chrom']}:{site['pos']} {site['ref']}>{site['alt']} {site['gt']}")

