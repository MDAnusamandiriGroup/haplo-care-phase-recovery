# Input data

The repository does not include the 1000 Genomes VCF or participant-level genomic data. To rerun the exploratory chromosome 22 analysis, obtain the files directly from the project source and keep them local in this folder.

- Phased high-coverage panel: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
- Sample and population metadata: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt

The script reads a 5 Mb GRCh38 region, chr22:30,000,000–34,999,999. Its labels are published statistical phase estimates, not independently verified molecular truth. See `docs/RESULTS_AND_INTERPRETATION.md` before interpreting the output.
