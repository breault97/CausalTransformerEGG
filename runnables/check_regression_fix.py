import sys
import os

# Make sure the src directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.physionet_eegmmidb.dataset import PhysioNetEEGMMIDBDatasetCollection

def run_check():
    """
    Checks if the PhysioNetEEGMMIDBDatasetCollection class has the _pre_scan_records method.
    """
    print("--- Running Regression Fix Check ---")
    if hasattr(PhysioNetEEGMMIDBDatasetCollection, "_pre_scan_records"):
        print("SUCCESS: `PhysioNetEEGMMIDBDatasetCollection` has the method `_pre_scan_records`.")
    else:
        print("FAILURE: `PhysioNetEEGMMIDBDatasetCollection` is MISSING the method `_pre_scan_records`.")
        sys.exit(1)
    
    # Check if it's a method (i.e., callable)
    if callable(getattr(PhysioNetEEGMMIDBDatasetCollection, "_pre_scan_records")):
         print("SUCCESS: `_pre_scan_records` is a callable method.")
    else:
        print("FAILURE: `_pre_scan_records` is not a callable method.")
        sys.exit(1)

    print("--- Check Passed ---")
    sys.exit(0)

if __name__ == "__main__":
    run_check()
