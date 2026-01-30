# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18455
- accuracy: 0.8274180438905445
- balanced_acc: 0.8227571826097052
- macro_f1: 0.8262211658480072
- mean_max_prob: 0.8839460611343384
- brier: 0.25408762748033686

## Record level
- samples: 308
- accuracy: 0.8701298701298701
- balanced_acc: 0.8702500472828055
- macro_f1: 0.8711807045135295
- mean_max_prob: 0.775651216506958
- brier: 0.2170432846744383

## Subject level
- samples: 22
- accuracy: 0.9090909090909091
- balanced_acc: 0.899999999999869
- macro_f1: 0.9074074074067728
- mean_max_prob: 0.7365732192993164
- brier: 0.1948026523153267

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.8711807045135295 | 0.8702500472828055 |
| csp_lda | 0.4533365376912601 | 0.5601212711818526 |
| simple_cnn | 0.738467532373584 | 0.7299789638591002 |
| shallowconv | 0.5467091993335088 | 0.5604353744681827 |
| eegnet | 0.7106677597596388 | 0.7065421764138673 |
