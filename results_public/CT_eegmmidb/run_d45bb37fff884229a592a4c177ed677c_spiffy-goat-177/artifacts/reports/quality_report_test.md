# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18455
- accuracy: 0.714332159306421
- balanced_acc: 0.7110880205087371
- macro_f1: 0.7119088460800752
- mean_max_prob: 0.7225624322891235
- brier: 0.3794013972917887

## Record level
- samples: 308
- accuracy: 0.7857142857142857
- balanced_acc: 0.7860432952969282
- macro_f1: 0.7841975297718813
- mean_max_prob: 0.6732155680656433
- brier: 0.31645608545467774

## Subject level
- samples: 22
- accuracy: 0.8636363636363636
- balanced_acc: 0.8857142857141515
- macro_f1: 0.8694463431299301
- mean_max_prob: 0.6498498320579529
- brier: 0.2763008122448959

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.7841975297718813 | 0.7860432952969282 |
| csp_lda | 0.4533365376912601 | 0.5601212711818526 |
| simple_cnn | 0.738467532373584 | 0.7299789638591002 |
| shallowconv | 0.5467091993335088 | 0.5604353744681827 |
| eegnet | 0.7106677597596388 | 0.7065421764138673 |
