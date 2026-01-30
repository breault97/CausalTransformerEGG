# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18455
- accuracy: 0.3511785424004335
- balanced_acc: 0.3333333333333333
- macro_f1: 0.17327023847703996
- mean_max_prob: 0.43237337470054626
- brier: 0.6718509886529783

## Record level
- samples: 308
- accuracy: 0.4155844155844156
- balanced_acc: 0.33333333333333076
- macro_f1: 0.1957186544341116
- mean_max_prob: 0.4323750138282776
- brier: 0.6537549561639685

## Subject level
- samples: 22
- accuracy: 0.45454545454545453
- balanced_acc: 0.3333333333333
- macro_f1: 0.20833333333317708
- mean_max_prob: 0.43237364292144775
- brier: 0.6423333720097659

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.1957186544341116 | 0.33333333333333076 |
| csp_lda | 0.4533365376912601 | 0.5601212711818526 |
| simple_cnn | 0.738467532373584 | 0.7299789638591002 |
| shallowconv | 0.5467091993335088 | 0.5604353744681827 |
| eegnet | 0.7106677597596388 | 0.7065421764138673 |
