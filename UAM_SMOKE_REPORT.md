# UAM Smoke Matrix Analysis

Each arm uses one seed and ten updates. This is a mechanism check; it supports no superiority claim.

## Provenance

Matrix root: `saves/exp/UAM_SMOKE/0712145130`

Manifest: **4/4** successful arms.

Checkpoint payloads: **0**.

## Method diagnostics

| Method | Mean effective ratio | Mean residual gate rate | Max kept residual orthogonality | Finalizers | Replays | Delta probes |
|---|---:|---:|---:|---:|---:|---:|
| uam_nll | 0.73810669 | 0 | NA | 10 | 10 | 2 |
| uam_simnpo | 0.72031853 | 0 | NA | 10 | 10 | 2 |
| uam_gu_nll | 0.97978383 | 0.4 | 3.5435149e-09 | 10 | 10 | 2 |
| uam_gu_simnpo | 0.97982001 | 0.2 | 1.4003124e-09 | 10 | 10 | 2 |

## Metric and actual-delta probes

| Method | Metric | Raw value | Delta vs uam_nll |
|---|---|---:|---:|
| uam_nll | exact_memorization | 0.93711246 | 0 |
| uam_nll | extraction_strength | 0.39998844 | 0 |
| uam_nll | forget_Q_A_Prob | 0.77953623 | 0 |
| uam_nll | forget_Q_A_ROUGE | 0.68803526 | 0 |
| uam_nll | forget_Q_A_gibberish | 0.89394078 | 0 |
| uam_nll | mia_gradnorm | 0.983125 | 0 |
| uam_nll | mia_loss | 0.985 | 0 |
| uam_nll | mia_min_k | 0.986875 | 0 |
| uam_nll | mia_min_k_plus_plus | 0.953125 | 0 |
| uam_nll | mia_zlib | 0.995 | 0 |
| uam_nll | model_utility | 0.59977039 | 0 |
| uam_nll | privleak | -97.520661 | 0 |
| uam_nll | retain_extraction_strength | 0.72652788 | 0 |
| uam_simnpo | exact_memorization | 0.93440222 | -0.0027102396 |
| uam_simnpo | extraction_strength | 0.38286624 | -0.017122197 |
| uam_simnpo | forget_Q_A_Prob | 0.76570623 | -0.013830006 |
| uam_simnpo | forget_Q_A_ROUGE | 0.61668112 | -0.071354138 |
| uam_simnpo | forget_Q_A_gibberish | 0.90973041 | 0.015789632 |
| uam_simnpo | mia_gradnorm | 0.979375 | -0.00375 |
| uam_simnpo | mia_loss | 0.985625 | 0.000625 |
| uam_simnpo | mia_min_k | 0.99 | 0.003125 |
| uam_simnpo | mia_min_k_plus_plus | 0.95125 | -0.001875 |
| uam_simnpo | mia_zlib | 0.996875 | 0.001875 |
| uam_simnpo | model_utility | 0.59886567 | -0.00090471971 |
| uam_simnpo | privleak | -98.11098 | -0.59031877 |
| uam_simnpo | retain_extraction_strength | 0.73205127 | 0.0055233908 |
| uam_gu_nll | exact_memorization | 0.93363988 | -0.0034725845 |
| uam_gu_nll | extraction_strength | 0.39743089 | -0.0025575527 |
| uam_gu_nll | forget_Q_A_Prob | 0.7701238 | -0.0094124369 |
| uam_gu_nll | forget_Q_A_ROUGE | 0.67904523 | -0.0089900281 |
| uam_gu_nll | forget_Q_A_gibberish | 0.92264107 | 0.028700287 |
| uam_gu_nll | mia_gradnorm | 0.985 | 0.001875 |
| uam_gu_nll | mia_loss | 0.981875 | -0.003125 |
| uam_gu_nll | mia_min_k | 0.985 | -0.001875 |
| uam_gu_nll | mia_min_k_plus_plus | 0.943125 | -0.01 |
| uam_gu_nll | mia_zlib | 0.993125 | -0.001875 |
| uam_gu_nll | model_utility | 0.59816359 | -0.0016068014 |
| uam_gu_nll | privleak | -97.16647 | 0.35419126 |
| uam_gu_nll | retain_extraction_strength | 0.72373573 | -0.0027921446 |
| uam_gu_simnpo | exact_memorization | 0.93114177 | -0.0059706986 |
| uam_gu_simnpo | extraction_strength | 0.39060631 | -0.0093821306 |
| uam_gu_simnpo | forget_Q_A_Prob | 0.75960643 | -0.019929806 |
| uam_gu_simnpo | forget_Q_A_ROUGE | 0.65024587 | -0.037789383 |
| uam_gu_simnpo | forget_Q_A_gibberish | 0.89468283 | 0.0007420525 |
| uam_gu_simnpo | mia_gradnorm | 0.984375 | 0.00125 |
| uam_gu_simnpo | mia_loss | 0.98375 | -0.00125 |
| uam_gu_simnpo | mia_min_k | 0.98625 | -0.000625 |
| uam_gu_simnpo | mia_min_k_plus_plus | 0.944375 | -0.00875 |
| uam_gu_simnpo | mia_zlib | 0.994375 | -0.000625 |
| uam_gu_simnpo | model_utility | 0.59870161 | -0.0010687749 |
| uam_gu_simnpo | privleak | -97.402597 | 0.11806375 |
| uam_gu_simnpo | retain_extraction_strength | 0.72324654 | -0.0032813328 |

| Method | Step | Coverage | Delta norm | Forget derivative | Retain derivative |
|---|---:|---|---:|---:|---:|
| uam_nll | 1 | full | 0.16953642 | 0.10686902 | -0.12239473 |
| uam_nll | 10 | full | 0.0069104473 | 0.0002902944 | -0.0042805925 |
| uam_simnpo | 1 | full | 0.16957313 | 0.079880235 | -0.10380281 |
| uam_simnpo | 10 | full | 0.0068243772 | 9.2627617e-05 | -0.003570067 |
| uam_gu_nll | 1 | full | 0.16957445 | 0.14791476 | -0.042023999 |
| uam_gu_nll | 10 | full | 0.007721753 | -0.00046452578 | -0.0057339002 |
| uam_gu_simnpo | 1 | full | 0.16963387 | 0.096459607 | -0.042707204 |
| uam_gu_simnpo | 10 | full | 0.0071079663 | -0.00014021027 | -0.0051506711 |
