# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 17640
- accuracy: 0.7979591836734694
- balanced_acc: 0.7839092575091565
- macro_f1: 0.7899374860996274
- mean_max_prob: 0.8369903564453125
- brier: 0.29382473910615015

## Record level
- samples: 294
- accuracy: 0.8537414965986394
- balanced_acc: 0.841948717948709
- macro_f1: 0.8505784226526408
- mean_max_prob: 0.715121328830719
- brier: 0.24068129055562076

## Subject level
- samples: 21
- accuracy: 0.8571428571428571
- balanced_acc: 0.8783068783067419
- macro_f1: 0.8745098039209354
- mean_max_prob: 0.6636689305305481
- brier: 0.2635353305190529

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.8505784226526408 | 0.841948717948709 |
| csp_lda | 0.28719968771070653 | 0.35714285714285715 |
| simple_cnn | 0.6915443745632425 | 0.683020683020683 |
| shallowconv | 0.4404845096668996 | 0.4654641654641655 |
| eegnet | 0.6308994780601526 | 0.6244829244829245 |
