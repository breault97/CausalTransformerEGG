# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18478
- accuracy: 0.33796947721614895
- balanced_acc: 0.3333333333333333
- macro_f1: 0.16839919642964937
- mean_max_prob: 0.4269833564758301
- brier: 0.6774082237278057

## Record level
- samples: 308
- accuracy: 0.36363636363636365
- balanced_acc: 0.3333333333333304
- macro_f1: 0.17777777777764656
- mean_max_prob: 0.42698317766189575
- brier: 0.6705235254160378

## Subject level
- samples: 22
- accuracy: 0.36363636363636365
- balanced_acc: 0.3333333333332917
- macro_f1: 0.17777777777763556
- mean_max_prob: 0.4269830882549286
- brier: 0.6706354291651885

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.17777777777764656 | 0.3333333333333304 |
| csp_lda | 0.4261715296198054 | 0.5529100529100529 |
| simple_cnn | 0.6903910611300849 | 0.7247023809523809 |
| shallowconv | 0.6312489822504478 | 0.660218253968254 |
| eegnet | 0.730961850527068 | 0.7602513227513228 |
