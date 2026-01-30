$ErrorActionPreference = "Stop"

# Set $env:EEGMMIDB_DIR if your data is not in the default config location.
$env_hint = 'Tip: $env:EEGMMIDB_DIR="C:\Projects\CausalTransformer\data\eegmmidb"'
Write-Host $env_hint
$configs = @(
  "eegmmidb_ct_baseline",
  "eegmmidb_ct_focal",
  "eegmmidb_ct_quick",
  "eegmmidb_ct_quantiles_20_60_20",
  "eegmmidb_ct_large",
  "eegmmidb_ct_large_focal"
)

foreach ($cfg in $configs) {
  Write-Host "=== $cfg: --cfg job ==="
  python runnables/train_multi.py +experiment=$cfg --cfg job
  Write-Host "=== $cfg: smoke (1 epoch) ==="
  python runnables/train_multi.py +experiment=$cfg exp.max_epochs=1
}
