# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18478
- accuracy: 0.8163762311938522
- balanced_acc: 0.8239574727289599
- macro_f1: 0.8241356620842512
- mean_max_prob: 0.8575606346130371
- brier: 0.26526709547186955

## Record level
- samples: 308
- accuracy: 0.8668831168831169
- balanced_acc: 0.8723756234349368
- macro_f1: 0.8743388863209502
- mean_max_prob: 0.721807599067688
- brier: 0.2332265167713373

## Subject level
- samples: 22
- accuracy: 0.9545454545454546
- balanced_acc: 0.9583333333331997
- macro_f1: 0.9521367521361231
- mean_max_prob: 0.6484265923500061
- brier: 0.2573260175783043

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.8743388863209502 | 0.8723756234349368 |
| csp_lda | 0.4261715296198054 | 0.5529100529100529 |
| simple_cnn | 0.6903910611300849 | 0.7247023809523809 |
| shallowconv | 0.6312489822504478 | 0.660218253968254 |
| eegnet | 0.730961850527068 | 0.7602513227513228 |
