# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 17640
- accuracy: 0.33015873015873015
- balanced_acc: 0.3333333333333333
- macro_f1: 0.16547334924410786
- mean_max_prob: 0.42829132080078125
- brier: 0.6843625505826341

## Record level
- samples: 294
- accuracy: 0.35374149659863946
- balanced_acc: 0.33333333333333015
- macro_f1: 0.17420435510874813
- mean_max_prob: 0.42829132080078125
- brier: 0.677764746690624

## Subject level
- samples: 21
- accuracy: 0.3333333333333333
- balanced_acc: 0.3333333333332857
- macro_f1: 0.16666666666652977
- mean_max_prob: 0.428291380405426
- brier: 0.6839983524779799

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.17420435510874813 | 0.33333333333333015 |
| csp_lda | 0.28719968771070653 | 0.35714285714285715 |
| simple_cnn | 0.6915443745632425 | 0.683020683020683 |
| shallowconv | 0.4404845096668996 | 0.4654641654641655 |
| eegnet | 0.6308994780601526 | 0.6244829244829245 |
