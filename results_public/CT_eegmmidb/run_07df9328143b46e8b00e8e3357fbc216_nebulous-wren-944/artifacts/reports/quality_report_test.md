# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 17640
- accuracy: 0.8047619047619048
- balanced_acc: 0.8037882527252843
- macro_f1: 0.7982317642440231
- mean_max_prob: 0.8670729994773865
- brier: 0.2835313709957541

## Record level
- samples: 294
- accuracy: 0.8401360544217688
- balanced_acc: 0.8505128205128111
- macro_f1: 0.8393640533683938
- mean_max_prob: 0.7383449673652649
- brier: 0.22903504481580983

## Subject level
- samples: 21
- accuracy: 0.8095238095238095
- balanced_acc: 0.8306878306877011
- macro_f1: 0.8156353450464903
- mean_max_prob: 0.6787167191505432
- brier: 0.24201400925479993

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.8393640533683938 | 0.8505128205128111 |
| csp_lda | 0.28719968771070653 | 0.35714285714285715 |
| simple_cnn | 0.6915443745632425 | 0.683020683020683 |
| shallowconv | 0.4404845096668996 | 0.4654641654641655 |
| eegnet | 0.6308994780601526 | 0.6244829244829245 |
