# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18479
- accuracy: 0.4227501488175767
- balanced_acc: 0.3333333333333333
- macro_f1: 0.19809060134632914
- mean_max_prob: 0.41953232884407043
- brier: 0.6539118461091061

## Record level
- samples: 308
- accuracy: 0.474025974025974
- balanced_acc: 0.3333333333333311
- macro_f1: 0.21439060205565386
- mean_max_prob: 0.4195324182510376
- brier: 0.6410608302590138

## Subject level
- samples: 22
- accuracy: 0.45454545454545453
- balanced_acc: 0.3333333333333
- macro_f1: 0.20833333333317708
- mean_max_prob: 0.4195324182510376
- brier: 0.6459565564059471

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.21439060205565386 | 0.3333333333333311 |
| csp_lda | 0.31770130454340983 | 0.3958553791887125 |
| simple_cnn | 0.8225358509881344 | 0.820695256660169 |
| shallowconv | 0.5351897449842408 | 0.5457904019307528 |
| eegnet | 0.7345202660493148 | 0.7358163928339367 |
