import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys

def analyze_predictions(run_dir: str):
    """
    Analyzes prediction files from an exported MLflow run.

    Args:
        run_dir: Path to the exported run directory (e.g., "mlflow_exports/.../run_<ID>_<NAME>").
    """
    print(f"Analyzing run directory: {run_dir}")

    # --- 1. Find prediction file ---
    predictions_dir = os.path.join(run_dir, "artifacts", "predictions")
    if not os.path.isdir(predictions_dir):
        # Fallback for older structures
        predictions_dir = os.path.join(run_dir, "artifacts")
        
    # Search for the most likely prediction file
    possible_files = [
        "predictions_test.csv",
        "test_predictions.csv",
    ]
    
    pred_file = None
    for f in possible_files:
        path = os.path.join(predictions_dir, f)
        if os.path.exists(path):
            pred_file = path
            break
            
    if pred_file is None:
        print(f"Error: Could not find a suitable test prediction CSV file in {predictions_dir}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found prediction file: {pred_file}")

    # --- 2. Load data ---
    try:
        df = pd.read_csv(pred_file)
    except Exception as e:
        print(f"Error: Failed to read CSV file {pred_file}. Reason: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Validate columns ---
    required_cols = ['y_true', 'y_pred']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Prediction file {pred_file} is missing one of the required columns: {required_cols}", file=sys.stderr)
        sys.exit(1)

    y_true = df['y_true']
    y_pred = df['y_pred']
    
    # Try to get class labels if they exist
    labels = sorted(df['y_true_label'].unique()) if 'y_true_label' in df.columns else None
    
    print(f"Analysis based on {len(df)} predictions.")

    # --- 4. Generate report ---
    report_str = classification_report(y_true, y_pred, target_names=labels, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    
    # Format confusion matrix for markdown
    cm_header = " | ".join([""] + (labels if labels else [f"Class {i}" for i in range(len(cm))]))
    cm_separator = "---|" * (len(cm) + 1)
    cm_rows = []
    for i, row in enumerate(cm):
        row_label = labels[i] if labels else f"Class {i}"
        cm_rows.append(f"{row_label} | " + " | ".join(map(str, row)))

    cm_markdown = f"| Predicted | {cm_header}\n|:{cm_separator}\n| **Actual** | {'<br>'.join(cm_rows)} |"


    # --- 5. Write markdown report ---
    report_path = os.path.join(run_dir, "artifacts", "analysis_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Analysis Report\n\n")
        f.write(f"Analysis of prediction file: `{os.path.basename(pred_file)}`\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(report_str)
        f.write("\n```\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write(f"```\n{cm}\n```\n\n")

    print(f"Successfully generated analysis report: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prediction files from an exported MLflow run.")
    parser.add_argument("run_dir", type=str, help="Path to the exported run directory.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: Provided path is not a valid directory: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    analyze_predictions(args.run_dir)
