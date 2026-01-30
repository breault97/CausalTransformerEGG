"""
Quick CLI check for EEGMMIDB dataset discovery.

Prints the resolved dataset root and a few example EDF paths using the same layout resolver
as the main dataset loader.
"""

import sys
from src.data.physionet_eegmmidb.dataset import summarize_eegmmidb_dir


def main():
    """Entry point: `python runnables/check_eegmmidb.py [DATA_DIR]`."""
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/mne"
    summarize_eegmmidb_dir(data_dir)


if __name__ == "__main__":
    main()
