# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18455
- accuracy: 0.8248171227309672
- balanced_acc: 0.8226138547317685
- macro_f1: 0.8245560884889152
- mean_max_prob: 0.8715014457702637
- brier: 0.2545980461805926

## Record level
- samples: 308
- accuracy: 0.8668831168831169
- balanced_acc: 0.8704634766527882
- macro_f1: 0.866638406826679
- mean_max_prob: 0.7715052366256714
- brier: 0.21412654185089858

## Subject level
- samples: 22
- accuracy: 0.9090909090909091
- balanced_acc: 0.899999999999869
- macro_f1: 0.899999999999369
- mean_max_prob: 0.7303197979927063
- brier: 0.20002142349168858

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.866638406826679 | 0.8704634766527882 |
| csp_lda | 0.4533365376912601 | 0.5601212711818526 |
| simple_cnn | 0.738467532373584 | 0.7299789638591002 |
| shallowconv | 0.5467091993335088 | 0.5604353744681827 |
| eegnet | 0.7106677597596388 | 0.7065421764138673 |
