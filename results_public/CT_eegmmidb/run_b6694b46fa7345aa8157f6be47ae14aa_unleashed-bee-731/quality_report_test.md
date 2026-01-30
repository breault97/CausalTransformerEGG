# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18479
- accuracy: 0.7427891119649331
- balanced_acc: 0.7751007067222856
- macro_f1: 0.7644866639184906
- mean_max_prob: 0.8329541683197021
- brier: 0.37241849601398497

## Record level
- samples: 308
- accuracy: 0.7564935064935064
- balanced_acc: 0.7687706894279337
- macro_f1: 0.7762116183163782
- mean_max_prob: 0.7560822367668152
- brier: 0.33687618621440285

## Subject level
- samples: 22
- accuracy: 0.7272727272727273
- balanced_acc: 0.7999999999997867
- macro_f1: 0.7916666666659681
- mean_max_prob: 0.7194583415985107
- brier: 0.3842234229604522

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.7762116183163782 | 0.7687706894279337 |
| csp_lda | 0.31770130454340983 | 0.3958553791887125 |
| simple_cnn | 0.8225358509881344 | 0.820695256660169 |
| shallowconv | 0.5351897449842408 | 0.5457904019307528 |
| eegnet | 0.7345202660493148 | 0.7358163928339367 |
