# PCGrad Multi-Dataset Smoke Analysis

Each arm uses one seed and ten optimizer updates. These are mechanism checks, not final performance claims.

## Provenance

Matrix root: `saves/exp/PCGRAD_SMOKE/0711204008`

Manifest: **17/17** successful arms.

Hydra configs: matched across control, GU, and PCGrad after excluding method-only fields.

Checkpoint payloads: **0**.

| Dataset | Conflict rate | Mean PCGrad–GU distance | Degenerate | Advance |
|---|---:|---:|:---:|:---:|
| tofu01 | 1 | 0 | yes | no |
| tofu05 | 0.8 | 8.926114e-06 | no | yes |
| tofu10 | 0.9 | 2.2556636e-05 | no | yes |
| muse_news | 1 | 0 | yes | no |
| muse_books | 0.8 | 1.0662361e-05 | no | yes |

Cross-dataset stop: **no**.

## tofu01

| Metric | Control | GU | PCGrad | GU−Control | PCGrad−Control | PCGrad−GU |
|---|---:|---:|---:|---:|---:|---:|
| exact_memorization | 0.94391604 | 0.94397661 | 0.94397661 | 6.057322e-05 | 6.057322e-05 | 0 |
| extraction_strength | 0.41234032 | 0.39770796 | 0.39770796 | -0.014632353 | -0.014632353 | 0 |
| forget_Q_A_Prob | 0.80246283 | 0.80286219 | 0.80286219 | 0.00039936155 | 0.00039936155 | 0 |
| forget_Q_A_ROUGE | 0.65090876 | 0.66116937 | 0.66116937 | 0.010260605 | 0.010260605 | 0 |
| forget_Q_A_gibberish | 0.93267394 | 0.92319048 | 0.92319048 | -0.0094834611 | -0.0094834611 | 0 |
| mia_gradnorm | 0.9596875 | 0.9584375 | 0.9584375 | -0.00125 | -0.00125 | 0 |
| mia_loss | 0.991875 | 0.991875 | 0.991875 | 0 | 0 | 0 |
| mia_min_k | 0.99375 | 0.99375 | 0.99375 | 0 | 0 | 0 |
| mia_min_k_plus_plus | 0.955625 | 0.95625 | 0.95625 | 0.000625 | 0.000625 | 0 |
| mia_zlib | 0.99875 | 0.99875 | 0.99875 | 0 | 0 | 0 |
| model_utility | 0.59763752 | 0.59693684 | 0.59693684 | -0.0007006752 | -0.0007006752 | 0 |
| privleak | -98.819362 | -98.819362 | -98.819362 | 0 | 0 | 0 |
| retain_extraction_strength | 0.719565 | 0.71737996 | 0.71737996 | -0.0021850354 | -0.0021850354 | 0 |

## tofu05

| Metric | Control | GU | PCGrad | GU−Control | PCGrad−Control | PCGrad−GU |
|---|---:|---:|---:|---:|---:|---:|
| exact_memorization | 0.96228998 | 0.96222754 | 0.96256036 | -6.2433183e-05 | 0.00027038604 | 0.00033281922 |
| extraction_strength | 0.6393428 | 0.6344302 | 0.64298954 | -0.0049125998 | 0.0036467381 | 0.008559338 |
| forget_Q_A_Prob | 0.86510488 | 0.86545273 | 0.86543253 | 0.0003478466 | 0.00032764614 | -2.0200461e-05 |
| forget_Q_A_ROUGE | 0.78681356 | 0.77793192 | 0.78190903 | -0.0088816363 | -0.0049045323 | 0.003977104 |
| forget_Q_A_gibberish | 0.86107887 | 0.87208863 | 0.86966494 | 0.011009763 | 0.0085860687 | -0.0024236944 |
| mia_gradnorm | 0.991975 | 0.991575 | 0.9920375 | -0.0004 | 6.25e-05 | 0.0004625 |
| mia_loss | 0.99985 | 0.999825 | 0.999875 | -2.5e-05 | 2.5e-05 | 5e-05 |
| mia_min_k | 0.99995 | 0.9999 | 0.999925 | -5e-05 | -2.5e-05 | 2.5e-05 |
| mia_min_k_plus_plus | 0.99885 | 0.9989 | 0.99875 | 5e-05 | -0.0001 | -0.00015 |
| mia_zlib | 0.999325 | 0.999375 | 0.99935 | 5e-05 | 2.5e-05 | -2.5e-05 |
| model_utility | 0.60194061 | 0.60000884 | 0.60166202 | -0.0019317731 | -0.00027858997 | 0.0016531831 |
| privleak | -99.992158 | -99.984316 | -99.988237 | 0.0078419072 | 0.0039209536 | -0.0039209536 |
| retain_extraction_strength | 0.71731025 | 0.71721304 | 0.71521655 | -9.7218695e-05 | -0.0020937082 | -0.0019964895 |

## tofu10

| Metric | Control | GU | PCGrad | GU−Control | PCGrad−Control | PCGrad−GU |
|---|---:|---:|---:|---:|---:|---:|
| exact_memorization | 0.96264376 | 0.96281097 | 0.96268295 | 0.00016720399 | 3.9191991e-05 | -0.000128012 |
| extraction_strength | 0.63751759 | 0.63752236 | 0.63904294 | 4.7688627e-06 | 0.0015253504 | 0.0015205815 |
| forget_Q_A_Prob | 0.86839027 | 0.86783655 | 0.86817895 | -0.00055372328 | -0.00021131665 | 0.00034240663 |
| forget_Q_A_ROUGE | 0.77174499 | 0.77436035 | 0.77570443 | 0.0026153564 | 0.0039594354 | 0.001344079 |
| forget_Q_A_gibberish | 0.8793399 | 0.87984462 | 0.87765125 | 0.00050471829 | -0.0016886474 | -0.0021933657 |
| mia_gradnorm | 0.990875 | 0.99045938 | 0.99139687 | -0.000415625 | 0.000521875 | 0.0009375 |
| mia_loss | 0.99825 | 0.99828125 | 0.9982625 | 3.125e-05 | 1.25e-05 | -1.875e-05 |
| mia_min_k | 0.99844375 | 0.99849375 | 0.998475 | 5e-05 | 3.125e-05 | -1.875e-05 |
| mia_min_k_plus_plus | 0.99743125 | 0.99745 | 0.9974125 | 1.875e-05 | -1.875e-05 | -3.75e-05 |
| mia_zlib | 0.9985875 | 0.99860625 | 0.99864375 | 1.875e-05 | 5.625e-05 | 3.75e-05 |
| model_utility | 0.59212227 | 0.59104653 | 0.59334785 | -0.0010757362 | 0.0012255806 | 0.0023013168 |
| privleak | -99.74793 | -99.756028 | -99.752991 | -0.0080986415 | -0.0050616509 | 0.0030369905 |
| retain_extraction_strength | 0.674024 | 0.67180606 | 0.66718472 | -0.0022179422 | -0.0068392744 | -0.0046213322 |

## muse_news

| Metric | Control | GU | PCGrad | GU−Control | PCGrad−Control | PCGrad−GU |
|---|---:|---:|---:|---:|---:|---:|
| exact_memorization | 0.94706354 | 0.94753972 | 0.94730163 | 0.00047618866 | 0.00023809493 | -0.00023809373 |
| extraction_strength | 0.30547619 | 0.30452381 | 0.30412698 | -0.00095238095 | -0.0013492063 | -0.0003968254 |
| forget_gibberish | 0.80555905 | 0.78625222 | 0.79234762 | -0.019306826 | -0.013211432 | 0.0060953942 |
| forget_knowmem_ROUGE | 0.63131198 | 0.6449334 | 0.64985376 | 0.013621418 | 0.018541777 | 0.0049203591 |
| forget_verbmem_ROUGE | 0.57775584 | 0.57727824 | 0.58126116 | -0.00047759971 | 0.0035053143 | 0.003982914 |
| mia_gradnorm | 0.9843 | 0.98265 | 0.98545 | -0.00165 | 0.00115 | 0.0028 |
| mia_loss | 0.999 | 0.999 | 0.999 | 0 | 0 | 0 |
| mia_min_k | 0.9992 | 0.9992 | 0.9992 | 0 | 0 | 0 |
| mia_min_k_plus_plus | 0.9859 | 0.9859 | 0.9838 | 0 | -0.0021 | -0.0021 |
| mia_reference | 1 | 1 | 1 | 0 | 0 | 0 |
| mia_zlib | 0.9998 | 0.9998 | 0.9998 | 0 | 0 | 0 |
| privleak | -99.832074 | -99.832074 | -99.832074 | 0 | 0 | 0 |
| retain_knowmem_ROUGE | 0.52583806 | 0.53668319 | 0.52550858 | 0.010845124 | -0.000329479 | -0.011174603 |

## muse_books

| Metric | Control | GU | PCGrad | GU−Control | PCGrad−Control | PCGrad−GU |
|---|---:|---:|---:|---:|---:|---:|
| exact_memorization | 0.99357145 | 0.99357145 | 0.99357145 | 0 | 0 | 0 |
| extraction_strength | 0.91595238 | 0.91595238 | 0.91595238 | 0 | 0 | 0 |
| forget_gibberish | 0.84124927 | 0.84124927 | 0.84124927 | 0 | 0 | 0 |
| forget_knowmem_ROUGE | 0.4730109 | 0.4730109 | 0.46326731 | 0 | -0.0097435897 | -0.0097435897 |
| forget_verbmem_ROUGE | 0.99701561 | 0.99701561 | 0.99701561 | 0 | 0 | 0 |
| mia_gradnorm | 0.7925 | 0.79885 | 0.79595 | 0.00635 | 0.00345 | -0.0029 |
| mia_loss | 0.7653 | 0.7655 | 0.7655 | 0.0002 | 0.0002 | 2.220446e-16 |
| mia_min_k | 0.7669 | 0.7667 | 0.7667 | -0.0002 | -0.0002 | 2.220446e-16 |
| mia_min_k_plus_plus | 0.7439 | 0.7436 | 0.7439 | -0.0003 | 0 | 0.0003 |
| mia_reference | 0.8036 | 0.8029 | 0.8035 | -0.0007 | -0.0001 | 0.0006 |
| mia_zlib | 0.7906 | 0.7905 | 0.7904 | -0.0001 | -0.0002 | -0.0001 |
| privleak | -56.897189 | -56.860207 | -56.860207 | 0.036982249 | 0.036982249 | -2.8421709e-14 |
| retain_knowmem_ROUGE | 0.67285464 | 0.66785464 | 0.67285464 | -0.005 | 0 | 0.005 |

## Systems isolation

| Arm | Conflict rate | Mean PCGrad–GU distance |
|---|---:|---:|
| tofu01/pcgrad/torch_eager_fp32 | 1 | 0 |
| tofu01/pcgrad/torch_flash | 1 | 0 |

### System metric comparison

| Arm | Metric | Value | Delta vs production |
|---|---|---:|---:|
| tofu01/pcgrad/torch_eager_fp32 | exact_memorization | 0.86395429 | -0.080022326 |
| tofu01/pcgrad/torch_eager_fp32 | extraction_strength | 0.17449213 | -0.22321583 |
| tofu01/pcgrad/torch_eager_fp32 | forget_Q_A_Prob | 0.54958206 | -0.25328012 |
| tofu01/pcgrad/torch_eager_fp32 | forget_Q_A_ROUGE | 0.46035547 | -0.2008139 |
| tofu01/pcgrad/torch_eager_fp32 | forget_Q_A_gibberish | 0.912148 | -0.011042488 |
| tofu01/pcgrad/torch_eager_fp32 | mia_gradnorm | 0.791875 | -0.1665625 |
| tofu01/pcgrad/torch_eager_fp32 | mia_loss | 0.960625 | -0.03125 |
| tofu01/pcgrad/torch_eager_fp32 | mia_min_k | 0.968125 | -0.025625 |
| tofu01/pcgrad/torch_eager_fp32 | mia_min_k_plus_plus | 0.805 | -0.15125 |
| tofu01/pcgrad/torch_eager_fp32 | mia_zlib | 0.979375 | -0.019375 |
| tofu01/pcgrad/torch_eager_fp32 | model_utility | 0.59810221 | 0.0011653673 |
| tofu01/pcgrad/torch_eager_fp32 | privleak | -93.978749 | 4.8406139 |
| tofu01/pcgrad/torch_eager_fp32 | retain_extraction_strength | 0.64105061 | -0.076329354 |
| tofu01/pcgrad/torch_flash | exact_memorization | 0.94639096 | 0.0024143472 |
| tofu01/pcgrad/torch_flash | extraction_strength | 0.41055626 | 0.012848297 |
| tofu01/pcgrad/torch_flash | forget_Q_A_Prob | 0.80177465 | -0.0010875329 |
| tofu01/pcgrad/torch_flash | forget_Q_A_ROUGE | 0.65514703 | -0.0060223397 |
| tofu01/pcgrad/torch_flash | forget_Q_A_gibberish | 0.93108833 | 0.0078978509 |
| tofu01/pcgrad/torch_flash | mia_gradnorm | 0.9590625 | 0.000625 |
| tofu01/pcgrad/torch_flash | mia_loss | 0.9925 | 0.000625 |
| tofu01/pcgrad/torch_flash | mia_min_k | 0.99375 | 0 |
| tofu01/pcgrad/torch_flash | mia_min_k_plus_plus | 0.956875 | 0.000625 |
| tofu01/pcgrad/torch_flash | mia_zlib | 0.99875 | 0 |
| tofu01/pcgrad/torch_flash | model_utility | 0.59628524 | -0.00065160421 |
| tofu01/pcgrad/torch_flash | privleak | -98.819362 | 0 |
| tofu01/pcgrad/torch_flash | retain_extraction_strength | 0.71889399 | 0.0015140298 |

## Actual update probes

| Arm | Step | Coverage | Delta norm | Forget dot | Retain dot |
|---|---:|---|---:|---:|---:|
| tofu01/pcgrad/production | 1 | full | 0 | 0 | 0 |
| tofu01/pcgrad/production | 10 | full | 0.0080112976 | -0.000380106 | -0.0063224049 |
| muse_news/gu/production | 1 | sampled | 0.0051261138 | -2.092458e-08 | -2.1694606e-05 |
| muse_news/gu/production | 10 | sampled | 0.00388074 | -8.4299176e-08 | -1.867337e-05 |
| muse_news/pcgrad/production | 1 | sampled | 0.0051254614 | -2.8432849e-08 | -2.169745e-05 |
| muse_news/pcgrad/production | 10 | sampled | 0.0053528518 | -2.1980926e-07 | -2.75481e-05 |
| muse_books/gu/production | 1 | sampled | 0.0046396762 | -4.2525026e-07 | -3.2901297e-06 |
| muse_books/gu/production | 10 | sampled | 0.0062643066 | -2.8040072e-07 | -7.6223046e-06 |
| muse_books/pcgrad/production | 1 | sampled | 0.0046399035 | -4.2559886e-07 | -3.2902597e-06 |
| muse_books/pcgrad/production | 10 | sampled | 0.006270913 | -2.7504138e-07 | -7.6723402e-06 |
| tofu01/pcgrad/torch_eager_fp32 | 1 | full | 0 | 0 | 0 |
| tofu01/pcgrad/torch_eager_fp32 | 10 | full | 0.038128065 | 0.0002619618 | -0.19293046 |
| tofu01/pcgrad/torch_flash | 1 | full | 0 | 0 | 0 |
| tofu01/pcgrad/torch_flash | 10 | full | 0.0080253407 | -0.0003744963 | -0.0063326845 |
