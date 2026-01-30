import pandas as pd
import json
import os

def analyze_runs():
    base_path = 'mlflow_exports/experiment_665075068361836799_CT_eegmmidb'
    summary_df = pd.read_csv(os.path.join(base_path, 'runs_summary.csv'))

    results = []

    for _, row in summary_df.iterrows():
        run_id = row['run_id']
        run_name = row['run_name']
        run_dir = os.path.join(base_path, f'run_{run_id}_{run_name}')
        
        variant = 'baseline' # Default variant
        fold = 'unknown'
        notes = []

        params_path = os.path.join(run_dir, 'params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                fold = params.get('fold_index', 'unknown')
                
                # Infer variant
                stratify = params.get('dataset/stratify_subject_split', 'True')
                class_weights_mode = params.get('exp/class_weights_mode', 'none')
                
                variant_parts = []
                if stratify == 'False':
                    variant_parts.append('nonstrat')
                if class_weights_mode == 'manual':
                    variant_parts.append('manualcw')
                
                if variant_parts:
                    variant = '_'.join(variant_parts)
                
        else:
            notes.append('params.json missing')

        # Check for predictions file
        predictions_path = os.path.join(run_dir, 'artifacts/predictions/predictions_test.csv')
        if not ('bold-ape-718' in run_name or os.path.exists(predictions_path)):
            notes.append('predictions_test.csv missing')

        # Metric extraction
        metric_rank = row.get('metric.multi_test_balanced_acc_record', float('nan'))
        macro_f1_record = row.get('metric.multi_test_f1_macro_record', float('nan'))
        bal_acc_record = row.get('metric.multi_test_balanced_acc_record', float('nan'))
        macro_f1_window = row.get('metric.multi_test_f1_macro', float('nan'))
        bal_acc_window = row.get('metric.multi_test_balanced_acc', float('nan'))

        results.append({
            'run_id': run_id,
            'run_name': run_name,
            'experiment/variant': variant,
            'fold': fold,
            'metric_rank': metric_rank,
            'macro_f1_record': macro_f1_record,
            'bal_acc_record': bal_acc_record,
            'macro_f1_window': macro_f1_window,
            'bal_acc_window': bal_acc_window,
            'notes': ', '.join(notes) if notes else ''
        })

    # Create and print markdown table
    md_table_header = "| run_id | run_name | experiment/variant | fold | metric_rank (bal_acc_record) | macro_f1_record | bal_acc_record | macro_f1_window | bal_acc_window | notes |\n"
    md_table_separator = "|---|---|---|---|---|---|---|---|---|---|
"
    
    # Sort results by metric_rank
    results.sort(key=lambda x: x['metric_rank'] if not pd.isna(x['metric_rank']) else -1, reverse=True)

    md_table_body = ""
    for res in results:
        md_table_body += f"| {res['run_id'][:8]}... | {res['run_name']} | {res['experiment/variant']} | {res['fold']} | {res['metric_rank']:.4f} | {res['macro_f1_record']:.4f} | {res['bal_acc_record']:.4f} | {res['macro_f1_window']:.4f} | {res['bal_acc_window']:.4f} | {res['notes']} |\n"
    
    print("## Tâche 1: Métriques Agrégées\n")
    print("La métrique de ranking `subject_balanced_acc` n'étant pas disponible, `multi_test_balanced_acc_record` est utilisée comme fallback.\n")
    print(md_table_header + md_table_separator + md_table_body)

    # Calculate and print global stats
    stats_df = pd.DataFrame([res for res in results if not pd.isna(res['metric_rank'])])
    if not stats_df.empty:
        stats_df['metric_rank'] = pd.to_numeric(stats_df['metric_rank'])
        
        print("\n## Statistiques Globales (sur `multi_test_balanced_acc_record`)\n")

        global_stats = stats_df['metric_rank'].describe()
        print(f"- **Overall:** Mean={global_stats['mean']:.4f}, Std={global_stats['std']:.4f}, Min={global_stats['min']:.4f}, Max={global_stats['max']:.4f}\n")

        # Per-variant stats
        print("### Statistiques par Variante\n")
        variant_stats = stats_df.groupby('experiment/variant')['metric_rank'].agg(['mean', 'std', 'min', 'max', 'count'])
        print(variant_stats.to_markdown())
    else:
        print("No runs with valid metrics found for statistics calculation.")


if __name__ == '__main__':
    analyze_runs()