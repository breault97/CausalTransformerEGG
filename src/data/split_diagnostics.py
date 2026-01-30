"""
Split diagnostics utilities (train/val/test composition reporting).

This helper is used by the EEGMMIDB pipeline to generate auditable summaries of:
- subject IDs present in each split,
- window-level and record-level label distributions per split.
"""

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def _get_subject_labels(labels: np.ndarray, subjects: np.ndarray) -> Dict[int, List[int]]:
    subject_labels = defaultdict(list)
    for label, subject in zip(labels, subjects):
        subject_labels[int(subject)].append(int(label))
    return subject_labels


def _get_subject_label_dist(subject_labels: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    return {
        subject: dict(Counter(labels))
        for subject, labels in subject_labels.items()
    }


def _get_agg_label_dist(labels: np.ndarray) -> Dict[int, int]:
    return dict(Counter(labels.tolist()))


def generate_split_diagnostics(
    output_dir: str,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    record_subjects: List[int],
    record_ids: List[int],
    window_labels: np.ndarray,
    record_labels: np.ndarray,
    label_names: List[str]
):
    """
    Generates and saves diagnostic reports about data splits.

    Args:
        output_dir: Directory to save the report files.
        train_indices: List of record indices for the training set.
        val_indices: List of record indices for the validation set.
        test_indices: List of record indices for the test set.
        record_subjects: List of subject IDs for each record.
        record_ids: List of unique IDs for each record.
        window_labels: Numpy array of window-level labels for all records.
        record_labels: Numpy array of record-level labels for all records.
        label_names: List of class names for the labels.
    """
    try:
        logger.info(f"Generating split diagnostics in: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        record_subjects = np.asarray(record_subjects)
        record_ids = np.asarray(record_ids)
        
        splits = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        full_report = {}

        for split_name, indices in splits.items():
            indices = np.asarray(indices)
            if len(indices) == 0:
                logger.warning(f"Split '{split_name}' is empty, skipping diagnostics for it.")
                continue

            split_subjects = record_subjects[indices]
            split_record_ids = record_ids[indices]
            split_window_labels = np.concatenate([window_labels[i] for i in indices])
            split_record_labels = record_labels[indices]
            
            unique_subjects = sorted(np.unique(split_subjects).tolist())

            report = {
                "num_records": len(indices),
                "num_subjects": len(unique_subjects),
                "subjects": unique_subjects,
                "window_label_distribution": {
                    label_names[int(k)]: int(v)
                    for k, v in _get_agg_label_dist(split_window_labels.flatten()).items()
                },
                "record_label_distribution": {
                    label_names[int(k)]: int(v)
                    for k, v in _get_agg_label_dist(split_record_labels).items()
                },
            }
            full_report[split_name] = report
        
        # Save JSON summary
        summary_path = os.path.join(output_dir, "split_summary.json")
        with open(summary_path, "w") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Split summary saved to {summary_path}")

        # Save subject lists to CSV
        subject_df_list = []
        for split_name, report_data in full_report.items():
            for subject_id in report_data["subjects"]:
                subject_df_list.append({"split": split_name, "subject_id": subject_id})
        
        if subject_df_list:
            subject_df = pd.DataFrame(subject_df_list)
            subjects_path = os.path.join(output_dir, "split_subjects.csv")
            subject_df.to_csv(subjects_path, index=False)
            logger.info(f"Split subjects saved to {subjects_path}")

    except Exception as e:
        logger.error(f"Failed to generate split diagnostics: {e}", exc_info=True)
