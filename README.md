# mismatch_genotype_model
Mismatch-based genotype prediction from read-level features in ancient DNA including feature extraction from BAM/VCF/FASTA and MLP classification.

## Overview
Code for genotype prediction from sequencing data using MLP and baseline models.

## Files
- `get_ref_base.py`: fetch reference base from FASTA
- `parse_one_site.py`: extract base counts from BAM
- `build_mismatch_matrix.py`: compute mismatch/substitution features

- `read_vcf_sites.py`: read and filter VCF sites
- `build_labeled_sites.py`: construct labeled dataset from VCF

- `prepare_training_data.py`: assemble feature matrix

- `train_baseline_model.py`: LR / baseline training
- `train_mlp_model.py`: MLP training

- `baseline_generalization_test.py`: baseline model generalization evaluation
- `wc_generalization_test.py`: weighted/class-balanced generalization evaluation

## Requirements
- Python 3.x
- pysam
- numpy
- pandas
- torch

## Usage
Run scripts individually, eg.:
```bash
python train_mlp_model.py
