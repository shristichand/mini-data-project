"""
scripts/download/run_download.py

Download and initial sampling script for FineWeb-2 Nepali Devanagari (npi_Deva).
Discovers remote shards on Hugging Face Hub, downloads the shard, inspects schema,
and creates the baseline 10,000-row sample.
"""

from mini_data_project.download.download_and_sample import main

if __name__ == "__main__":
    main()
