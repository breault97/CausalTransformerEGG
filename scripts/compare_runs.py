import argparse
import os
import json
import pandas as pd
import numpy as np
from collections import Counter

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def generate_report(run_dir_a, run_dir_b, output_path):
    report_lines = []

    def process_run(run_dir, run_name):
        report_lines.append(f'## Analysis for Run: {run_name} (`{run_dir}`)')
        
        # 1. Load split summary
        split_summary_path = os.path.join(run_dir, 'artifacts', 'splits', 'split_summary.json')
        split_summary = load_json(split_summary_path)
        if split_summary and 'test' in split_summary:
            report_lines.append('### Test Set Composition')
            test_split_info = split_summary['test']
            report_lines.append(f"- **Subjects:** {test_split_info.get('num_subjects', 'N/A')}")
            report_lines.append(f"- **Records:** {test_split_info.get('num_records', 'N/A')}")
            report_lines.append('#### Label Distribution (Record Level)')
            for label, count in test_split_info.get('record_label_distribution', {}).items():
                report_lines.append(f'- {label}: {count}')
            report_lines.append("")
        else:
            report_lines.append("- _Split summary not found._\n")

        # 2. Load predictions
        preds_path = os.path.join(run_dir, 'predictions_test.csv')
        preds_df = load_csv(preds_path)
        if preds_df is None:
            report_lines.append("- _`predictions_test.csv` not found._\n")
            return None

        report_lines.append("### Prediction Analysis (Window Level)")
        
        # Confusion matrix
        y_true = preds_df['y_true_label']
        y_pred = preds_df['y_pred_label']
        confusion = pd.crosstab(y_true, y_pred)
        report_lines.append("#### Confusion Matrix")
        report_lines.append(confusion.to_markdown())
        report_lines.append("")

        # Top confusions
        confusions = Counter()
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                confusions[(true_label, pred_label)] += 1
        
        report_lines.append("#### Top Confusions (True -> Pred)")
        for (true, pred), count in confusions.most_common(5):
            report_lines.append(f"- {true} -> {pred}: {count} times")
        report_lines.append("")
        
        # Subject-level analysis
        if 'subject_id' in preds_df.columns:
            subject_errors = preds_df[preds_df['y_true'] != preds_df['y_pred']].groupby('subject_id').size()
            subject_counts = preds_df.groupby('subject_id').size()
            subject_error_rate = (subject_errors / subject_counts).fillna(0).sort_values(ascending=False)
            
            report_lines.append("#### Top 5 Subjects by Error Rate")
            for subject_id, rate in subject_error_rate.head(5).items():
                errors = subject_errors.get(subject_id, 0)
                total = subject_counts.get(subject_id, 0)
                report_lines.append(f"- **Subject {subject_id}:** {rate:.2%} error rate ({errors}/{total} windows)")
            report_lines.append("")
        
        return preds_df

    df_a = process_run(run_dir_a, 'Run A')
    df_b = process_run(run_dir_b, 'Run B')

    report_lines.insert(0, f'# Comparison Report: Run A vs Run B')
    report_lines.insert(1, f"- **Run A:** `{run_dir_a}`")
    report_lines.insert(2, f"- **Run B:** `{run_dir_b}`")
    report_lines.insert(3, "")

    if df_a is not None and df_b is not None:
        report_lines.append("## Overall Performance Comparison")
        acc_a = (df_a['y_true'] == df_a['y_pred']).mean()
        acc_b = (df_b['y_true'] == df_b['y_pred']).mean()
        report_lines.append(f"- **Accuracy A:** {acc_a:.4f}")
        report_lines.append(f"- **Accuracy B:** {acc_b:.4f}")
        report_lines.append(f"- **Difference (B - A):** {acc_b - acc_a:+.4f}")
        report_lines.append("")

    report_lines.append("## Hypotheses on Variance")
    report_lines.append("- **Distributional Shift:** Check if the subject and label distributions in the test sets differ significantly between runs. A run with more 'difficult' subjects or a higher proportion of a hard-to-classify class may perform worse.")
    report_lines.append("- **Subject-Specific Difficulty:** Identify if the same subjects are consistently misclassified across both runs. Some subjects might be inherently harder to model regardless of the fold.")
    report_lines.append("- **Model Instability:** If the distributions are similar but performance varies wildly, it could point to model instability (e.g., sensitivity to initialization or training dynamics).")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Comparison report generated at: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two MLflow runs.')
    parser.add_argument('run_dir_a', type=str, help='Path to the first run directory (e.g., mlflow_exports/.../run_A).')
    parser.add_argument('run_dir_b', type=str, help='Path to the second run directory (e.g., mlflow_exports/.../run_B).')
    parser.add_argument('--output', '-o', type=str, default='compare_report.md', help='Path to save the output markdown report.')
    
    args = parser.parse_args()
    
    # Create a default output in the directory of the first run if not specified
    if args.output == 'compare_report.md':
        output_dir = os.path.join(args.run_dir_a, 'artifacts')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'compare_report.md')
    else:
        output_path = args.output

    generate_report(args.run_dir_a, args.run_dir_b, output_path)
