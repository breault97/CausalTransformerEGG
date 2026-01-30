import argparse
import os
from collections import Counter


def _find_edf_files(data_dir):
    files = []
    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            if f.lower().endswith('.edf'):
                files.append(os.path.join(root, f))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Inspect WFDB event annotations for EEGMMIDB")
    parser.add_argument("--data-dir", default=os.path.join(os.getcwd(), "data", "eegmmidb"),
                        help="Root directory to search for .edf files")
    parser.add_argument("--n", type=int, default=50, help="Number of records to inspect")
    args = parser.parse_args()

    try:
        import wfdb
    except Exception as e:
        raise ImportError("wfdb is required. Install with: pip install wfdb") from e

    edf_files = _find_edf_files(args.data_dir)
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found under {args.data_dir}")

    symbol_counter = Counter()
    aux_counter = Counter()

    sample_dump = None
    inspected = 0
    for edf_path in edf_files:
        if inspected >= args.n:
            break
        try:
            ann = wfdb.rdann(edf_path, "event")
        except Exception:
            continue

        symbols = []
        aux_notes = []
        try:
            symbols = [str(s) for s in list(getattr(ann, "symbol", []))]
        except Exception:
            symbols = []
        try:
            aux_notes = [str(a) for a in list(getattr(ann, "aux_note", []))]
        except Exception:
            aux_notes = []

        if symbols:
            symbol_counter.update(symbols)
        if aux_notes:
            aux_counter.update(aux_notes)

        if sample_dump is None:
            samples = list(getattr(ann, "sample", []))
            if samples:
                rows = []
                for i in range(min(15, len(samples))):
                    s = samples[i]
                    sym = symbols[i] if i < len(symbols) else ""
                    aux = aux_notes[i] if i < len(aux_notes) else ""
                    rows.append((int(s), sym, aux))
                if rows:
                    sample_dump = (edf_path, rows)

        inspected += 1

    print(f"Inspected records: {inspected}")
    print("Top 20 ann.symbol counts:")
    for sym, cnt in symbol_counter.most_common(20):
        print(f"  {sym}: {cnt}")
    print("Top 20 ann.aux_note counts:")
    for aux, cnt in aux_counter.most_common(20):
        print(f"  {aux}: {cnt}")

    if sample_dump is not None:
        path, rows = sample_dump
        print(f"\nSample events from: {path}")
        for s, sym, aux in rows:
            print(f"  sample={s} symbol={sym} aux_note={aux}")


if __name__ == "__main__":
    main()
