# Supplementary Material

### Why Network Intrusion Detectors Fail Across Years: Decomposing Cross-Year Failure into Covariate Shift, Concept Change, and Prior-Probability Shift

_LightGBM RF-mode_

---
## Scope

- **Dataset:** CIC-IDS 2017 and 2018, corrected 2022 re-extraction.
- **Model:** LightGBM RF-mode tree ensemble (RandomForest variant).
- **Scope of claims:** tree-ensemble NIDS only; results do not generalize to neural nets, SVMs, or other dataset pairs.
- **Confound control:** both years use the same 2022-corrected extraction, so measured shift reflects network/attacker/time differences, not extractor artifacts.

---
## Step 0: Initial Data Exploration

Before any cleaning, looked at the raw per-year CSV files (a separate file per capture day) to know what is actually being worked with: column inventory, row counts, label distribution, and data-quality flags (NaN/Inf rows). Followed the original dataset researchers' methodology: rows whose raw label carries an explicit "- Attempted" suffix (an attempted-but-failed attack, which produces benign-shaped traffic) are folded into Benign during cleaning (step 1) rather than kept as a separate attack family.

Found 91 raw columns and 2,099,976 total rows across 2017's capture files, 91 columns and 63,195,145 rows across 2018's capture files.

| Label (2017, after attempted→Benign fold) | Count | % |
|---|---:|---:|
| Benign | 1,594,545 | 75.93% |
| Portscan | 159,066 | 7.57% |
| DoS Hulk | 158,468 | 7.55% |
| DDoS | 95,144 | 4.53% |
| Infiltration - Portscan | 71,767 | 3.42% |
| DoS GoldenEye | 7,567 | 0.36% |
| FTP-Patator | 3,972 | 0.19% |
| DoS Slowloris | 3,859 | 0.18% |
| SSH-Patator | 2,961 | 0.14% |
| DoS Slowhttptest | 1,740 | 0.08% |

| Label (2018, after attempted→Benign fold) | Count | % |
|---|---:|---:|
| Benign | 59,659,723 | 94.41% |
| DoS Hulk | 1,803,160 | 2.85% |
| DDoS-HOIC | 1,082,293 | 1.71% |
| DDoS-LOIC-HTTP | 289,328 | 0.46% |
| Botnet Ares | 142,921 | 0.23% |
| SSH-BruteForce | 94,197 | 0.15% |
| Infiltration - NMAP Portscan | 89,374 | 0.14% |
| DoS GoldenEye | 22,560 | 0.04% |
| DoS Slowloris | 8,490 | 0.01% |
| DDoS-LOIC-UDP | 2,527 | 0.00% |


The two tables above show the labels AFTER the attempted→Benign fold. The two below show the same rows BEFORE that fold, so each attempted-but-failed attack appears as its own row with the original attack type visible in the label text (e.g. `Web Attack - Brute Force - Attempted`, `Infiltration - Attempted`).

| Label (2017, before attempted→Benign fold) | Count | % |
|---|---:|---:|
| Benign | 1,582,566 | 75.36% |
| Portscan | 159,066 | 7.57% |
| DoS Hulk | 158,468 | 7.55% |
| DDoS | 95,144 | 4.53% |
| Infiltration - Portscan | 71,767 | 3.42% |
| DoS GoldenEye | 7,567 | 0.36% |
| Botnet - Attempted | 4,067 | 0.19% |
| FTP-Patator | 3,972 | 0.19% |
| DoS Slowloris | 3,859 | 0.18% |
| DoS Slowhttptest - Attempted | 3,368 | 0.16% |
| SSH-Patator | 2,961 | 0.14% |
| DoS Slowloris - Attempted | 1,847 | 0.09% |
| DoS Slowhttptest | 1,740 | 0.08% |
| Web Attack - Brute Force - Attempted | 1,292 | 0.06% |
| Botnet | 736 | 0.04% |
| Web Attack - XSS - Attempted | 655 | 0.03% |
| DoS Hulk - Attempted | 581 | 0.03% |
| DoS GoldenEye - Attempted | 80 | 0.00% |
| Web Attack - Brute Force | 73 | 0.00% |
| Infiltration - Attempted | 45 | 0.00% |
| Infiltration | 36 | 0.00% |
| SSH-Patator - Attempted | 27 | 0.00% |
| Web Attack - XSS | 18 | 0.00% |
| Web Attack - SQL Injection | 13 | 0.00% |
| FTP-Patator - Attempted | 12 | 0.00% |
| Heartbleed | 11 | 0.00% |
| Web Attack - SQL Injection - Attempted | 5 | 0.00% |

| Label (2018, before attempted→Benign fold) | Count | % |
|---|---:|---:|
| Benign | 59,353,486 | 93.92% |
| DoS Hulk | 1,803,160 | 2.85% |
| DDoS-HOIC | 1,082,293 | 1.71% |
| FTP-BruteForce - Attempted | 298,874 | 0.47% |
| DDoS-LOIC-HTTP | 289,328 | 0.46% |
| Botnet Ares | 142,921 | 0.23% |
| SSH-BruteForce | 94,197 | 0.15% |
| Infiltration - NMAP Portscan | 89,374 | 0.14% |
| DoS GoldenEye | 22,560 | 0.04% |
| DoS Slowloris | 8,490 | 0.01% |
| DoS GoldenEye - Attempted | 4,301 | 0.01% |
| DDoS-LOIC-UDP | 2,527 | 0.00% |
| DoS Slowloris - Attempted | 2,280 | 0.00% |
| Botnet Ares - Attempted | 262 | 0.00% |
| DDoS-LOIC-UDP - Attempted | 251 | 0.00% |
| Infiltration - Communication Victim Attacker | 204 | 0.00% |
| Web Attack - Brute Force - Attempted | 137 | 0.00% |
| Web Attack - Brute Force | 131 | 0.00% |
| Web Attack - XSS | 113 | 0.00% |
| DoS Hulk - Attempted | 86 | 0.00% |
| Infiltration - Dropbox Download | 85 | 0.00% |
| Web Attack - SQL | 39 | 0.00% |
| Infiltration - Dropbox Download - Attempted | 28 | 0.00% |
| Web Attack - SQL - Attempted | 14 | 0.00% |
| Web Attack - XSS - Attempted | 4 | 0.00% |


![2017: labels per file](results/0_dataexplore/cicids2017/2_files_per_label.png)

![2017: data quality per file](results/0_dataexplore/cicids2017/3_quality_per_file.png)

![2018: labels per file](results/0_dataexplore/cicids2018/2_files_per_label.png)

![2018: data quality per file](results/0_dataexplore/cicids2018/3_quality_per_file.png)

Full reports: `results/0_dataexplore/cicids2017/0_dataexplore_report.txt` and `results/0_dataexplore/cicids2018/0_dataexplore_report.txt`.

---
## Step 1: Loading, Cleaning, and Combining

Combined that year's raw capture-day CSVs into one file, dropped rows with NaN or Inf values, removed exact duplicate rows (row-hash based), and consolidated raw labels into the canonical attack-family set every later step trains and tests on. Applied a class-count floor: any canonical family with fewer than 100 rows was dropped as too sparse to model reliably.

### cicids2017: row-count funnel

| Stage | Rows |
|---|---:|
| Total raw rows | 2,099,976 |
| NaN rows removed | 0 |
| Inf rows removed | 5 |
| After NaN/Inf clean | 2,099,971 |
| Duplicates removed | 0 |
| After global dedup | 2,099,971 |
| Final (post-labels) | 2,099,960 |

### Canonical families (after consolidation)

| Family | Count |
|---|---:|
| Benign | 1,594,540 |
| DoS | 171,634 |
| PortScan | 159,066 |
| DDoS | 95,144 |
| Infiltration | 71,803 |
| BruteForce | 6,933 |
| Botnet | 736 |
| WebAttack | 104 |

Smallest canonical family: `WebAttack` at 104 rows, above the 100-row floor, so nothing was dropped at this stage in this run.

### Label consolidation: raw labels → canonical families

Raw CSV labels are consolidated into the canonical family set shown above. Examples of the mapping:

| Raw label | Canonical family |
|---|---|
| Benign | Benign |
| DoS Hulk | DoS |
| DoS GoldenEye | DoS |
| DoS Slowloris | DoS |
| DDoS HOIC | DDoS |
| LOIC HTTP | DDoS |
| LOIC UDP | DDoS |
| SSH-Patator | Brute Force |
| FTP-Patator | Brute Force |
| Web Attack – Brute Force | Web Attack |
| Web Attack – XSS | Web Attack |
| Infiltration | Infiltration |



### cicids2018: row-count funnel

| Stage | Rows |
|---|---:|
| Total raw rows | 63,195,145 |
| NaN rows removed | 0 |
| Inf rows removed | 57 |
| After NaN/Inf clean | 63,195,088 |
| Duplicates removed | 0 |
| After global dedup | 63,195,088 |
| Final (post-labels) | 63,195,088 |

### Canonical families (after consolidation)

| Family | Count |
|---|---:|
| Benign | 59,659,666 |
| DoS | 1,834,210 |
| DDoS | 1,374,148 |
| Botnet | 142,921 |
| BruteForce | 94,197 |
| Infiltration | 89,663 |
| WebAttack | 283 |

Smallest canonical family: `WebAttack` at 283 rows, above the 100-row floor, so nothing was dropped at this stage in this run.

### Label consolidation: raw labels → canonical families

Raw CSV labels are consolidated into the canonical family set shown above. Examples of the mapping:

| Raw label | Canonical family |
|---|---|
| Benign | Benign |
| DoS Hulk | DoS |
| DoS GoldenEye | DoS |
| DoS Slowloris | DoS |
| DDoS HOIC | DDoS |
| LOIC HTTP | DDoS |
| LOIC UDP | DDoS |
| SSH-Patator | Brute Force |
| FTP-Patator | Brute Force |
| Web Attack – Brute Force | Web Attack |
| Web Attack – XSS | Web Attack |
| Infiltration | Infiltration |



![2017: labels, raw](results/1_load_clean_combine/cicids2017/labels_stage1_raw.png)

![2017: labels, final (post-dedup, post-consolidation)](results/1_load_clean_combine/cicids2017/labels_stage4_final.png)

![2018: labels, raw](results/1_load_clean_combine/cicids2018/labels_stage1_raw.png)

![2018: labels, final (post-dedup, post-consolidation)](results/1_load_clean_combine/cicids2018/labels_stage4_final.png)

Output: `data/cc_data/cicids2017_cleaned.parquet` and `data/cc_data/cicids2018_cleaned.parquet`, the input every later step reads from.

---
## Step 2: Per-Year Correlation Matrices

For each year independently, computed Pearson (exact) and Spearman (deterministic subsample) correlation between every candidate feature pair, then flagged any pair at |r| >= 0.9 in either metric. This step only surfaces candidates; it does not decide what to drop (step 3 does that, next).

### cicids2017: pairs with Pearson or Spearman |r| >= 0.95 (of 81 flagged at the 0.9 screen)

| Feature A | Feature B | Pearson r | Spearman r |
|---|---|---:|---:|
| Fwd Packet Length Mean | Fwd Segment Size Avg | 1.0000 | 1.0000 |
| Bwd Packet Length Mean | Bwd Segment Size Avg | 1.0000 | 1.0000 |
| Packet Length Mean | Average Packet Size | 1.0000 | 1.0000 |
| ICMP Code | ICMP Type | 0.7658 | 1.0000 |
| Packet Length Std | Packet Length Variance | 0.9394 | 1.0000 |
| Total Bwd packets | ACK Flag Count | 0.9998 | 0.7873 |
| Fwd Bytes/Bulk Avg | Fwd Packet/Bulk Avg | 0.8574 | 0.9998 |
| Fwd Bytes/Bulk Avg | Fwd Bulk Rate Avg | 0.0591 | 0.9997 |
| Fwd Packet/Bulk Avg | Fwd Bulk Rate Avg | 0.1127 | 0.9997 |
| Total Fwd Packet | ACK Flag Count | 0.9997 | 0.7839 |
| Fwd URG Flags | URG Flag Count | 0.9996 | 0.9947 |
| Flow Duration | Fwd IAT Total | 0.9995 | 0.8070 |
| Idle Mean | Idle Max | 0.9870 | 0.9991 |
| Total Fwd Packet | Total Bwd packets | 0.9991 | 0.9317 |
| Flow Packets/s | Fwd Packets/s | 0.9690 | 0.9989 |
| Fwd Packet Length Min | Packet Length Min | 0.8395 | 0.9988 |
| Flow IAT Max | Fwd IAT Max | 0.9986 | 0.7706 |
| Flow IAT Max | Idle Max | 0.9983 | 0.7279 |
| Idle Mean | Idle Min | 0.9896 | 0.9978 |
| Active Mean | Active Max | 0.8350 | 0.9978 |
| Bwd Bytes/Bulk Avg | Bwd Packet/Bulk Avg | 0.9790 | 0.9972 |
| Total Fwd Packet | Total Length of Bwd Packet | 0.9970 | 0.7439 |
| Fwd IAT Max | Idle Max | 0.9969 | 0.7294 |
| Bwd IAT Total | Bwd IAT Max | 0.6600 | 0.9968 |
| Flow Packets/s | Flow IAT Mean | -0.0407 | -0.9960 |
| Fwd PSH Flags | PSH Flag Count | 0.4578 | 0.9958 |
| Total Length of Bwd Packet | ACK Flag Count | 0.9957 | 0.5601 |
| Flow IAT Mean | Fwd Packets/s | -0.0327 | -0.9957 |
| Idle Max | Idle Min | 0.9558 | 0.9956 |
| Fwd IAT Total | Fwd IAT Max | 0.6612 | 0.9955 |
| Bwd Segment Size Avg | Subflow Bwd Bytes | 0.9810 | 0.9954 |
| Bwd Packet Length Mean | Subflow Bwd Bytes | 0.9810 | 0.9954 |
| Bwd IAT Mean | Bwd IAT Max | 0.7841 | 0.9949 |
| Total Bwd packets | Total Length of Bwd Packet | 0.9944 | 0.8056 |
| Active Mean | Active Min | 0.9070 | 0.9942 |
| Fwd IAT Mean | Fwd IAT Max | 0.7578 | 0.9939 |
| Bwd Bytes/Bulk Avg | Bwd Bulk Rate Avg | 0.0105 | 0.9933 |
| Bwd IAT Total | Bwd IAT Mean | 0.4538 | 0.9933 |
| Fwd IAT Total | Fwd IAT Mean | 0.4335 | 0.9927 |
| Packet Length Max | Packet Length Variance | 0.9339 | 0.9898 |
| Packet Length Max | Packet Length Std | 0.9883 | 0.9898 |
| Bwd Packet/Bulk Avg | Bwd Bulk Rate Avg | 0.0105 | 0.9894 |
| Bwd Packet Length Max | Packet Length Max | 0.9894 | 0.9856 |
| Active Max | Active Min | 0.5908 | 0.9885 |
| Average Packet Size | Subflow Bwd Bytes | 0.9877 | 0.9711 |
| Packet Length Mean | Subflow Bwd Bytes | 0.9877 | 0.9711 |
| Bwd PSH Flags | PSH Flag Count | 0.9556 | 0.9863 |
| Flow Duration | Flow IAT Max | 0.6605 | 0.9860 |
| Flow IAT Max | Idle Mean | 0.9853 | 0.7273 |
| Fwd IAT Max | Idle Mean | 0.9840 | 0.7290 |
| Bwd Packet Length Max | Bwd Packet Length Std | 0.9836 | 0.8254 |
| Bwd Packet Length Max | Packet Length Std | 0.9834 | 0.9799 |
| Flow Duration | Bwd IAT Total | 0.9828 | 0.7842 |
| Fwd IAT Total | Bwd IAT Total | 0.9822 | 0.9186 |
| Bwd Packet Length Std | Packet Length Std | 0.9821 | 0.8316 |
| Active Std | Idle Std | 0.2425 | 0.9807 |
| Fwd Packet Length Mean | Subflow Fwd Bytes | 0.9747 | 0.9806 |
| Fwd Segment Size Avg | Subflow Fwd Bytes | 0.9747 | 0.9806 |
| Bwd Packet Length Max | Packet Length Variance | 0.9333 | 0.9799 |
| Fwd PSH Flags | Bwd PSH Flags | 0.1756 | 0.9744 |
| Bwd Packet Length Std | Packet Length Max | 0.9729 | 0.8276 |
| Flow Packets/s | Flow IAT Max | -0.0616 | -0.9723 |
| Flow IAT Max | Fwd Packets/s | -0.0496 | -0.9710 |
| Average Packet Size | Bwd Segment Size Avg | 0.9677 | 0.9700 |
| Packet Length Mean | Bwd Segment Size Avg | 0.9677 | 0.9700 |
| Bwd Packet Length Mean | Packet Length Mean | 0.9677 | 0.9700 |
| Bwd Packet Length Mean | Average Packet Size | 0.9677 | 0.9700 |
| Active Max | Idle Max | 0.2055 | 0.9687 |
| Active Max | Idle Mean | 0.1701 | 0.9686 |
| Active Min | Idle Min | 0.1432 | 0.9683 |
| Active Mean | Idle Min | 0.1575 | 0.9678 |
| Active Mean | Idle Mean | 0.1856 | 0.9676 |
| Active Max | Idle Min | 0.1375 | 0.9671 |
| Active Mean | Idle Max | 0.2183 | 0.9671 |
| Fwd IAT Mean | Fwd IAT Min | 0.9669 | 0.7220 |
| Active Min | Idle Mean | 0.1577 | 0.9655 |
| Bwd Packet Length Std | Packet Length Variance | 0.9640 | 0.8316 |
| Active Min | Idle Max | 0.1720 | 0.9640 |
| Bwd Packet Length Mean | Packet Length Std | 0.9635 | 0.9331 |
| Packet Length Std | Bwd Segment Size Avg | 0.9635 | 0.9331 |
| Total Length of Bwd Packet | Bwd Packet Length Max | 0.0207 | 0.9603 |
| Flow IAT Max | Bwd IAT Max | 0.9594 | 0.7427 |
| Packet Length Max | Packet Length Mean | 0.9128 | 0.9588 |
| Packet Length Max | Average Packet Size | 0.9128 | 0.9588 |
| Flow Duration | Flow Packets/s | -0.0725 | -0.9587 |
| Bwd IAT Max | Idle Max | 0.9580 | 0.6758 |
| Fwd IAT Max | Bwd IAT Max | 0.9579 | 0.9136 |
| Flow IAT Mean | Flow IAT Max | 0.7285 | 0.9573 |
| Flow Duration | Fwd Packets/s | -0.0583 | -0.9570 |
| Bwd IAT Mean | Bwd IAT Min | 0.9549 | 0.7344 |
| Bwd Packet Length Max | Bwd Packet Length Mean | 0.9505 | 0.9547 |
| Bwd Packet Length Max | Bwd Segment Size Avg | 0.9505 | 0.9547 |
| Flow IAT Max | Idle Min | 0.9542 | 0.7247 |
| Fwd IAT Max | Idle Min | 0.9529 | 0.7265 |
| Bwd IAT Max | Idle Mean | 0.9519 | 0.6758 |
| Bwd Packet Length Max | Packet Length Mean | 0.9081 | 0.9514 |
| Bwd Packet Length Max | Average Packet Size | 0.9081 | 0.9514 |
| Packet Length Mean | Packet Length Variance | 0.8016 | 0.9508 |
| Packet Length Variance | Average Packet Size | 0.8016 | 0.9508 |
| Packet Length Mean | Packet Length Std | 0.9352 | 0.9507 |
| Packet Length Std | Average Packet Size | 0.9352 | 0.9507 |
| Fwd IAT Min | Bwd IAT Min | 0.9502 | 0.5947 |


### cicids2018: pairs with Pearson or Spearman |r| >= 0.95 (of 47 flagged at the 0.9 screen)

| Feature A | Feature B | Pearson r | Spearman r |
|---|---|---:|---:|
| Bwd Packet Length Mean | Bwd Segment Size Avg | 1.0000 | 1.0000 |
| Fwd URG Flags | URG Flag Count | 1.0000 | 1.0000 |
| Packet Length Mean | Average Packet Size | 1.0000 | 1.0000 |
| Fwd Packet Length Mean | Fwd Segment Size Avg | 1.0000 | 1.0000 |
| Packet Length Std | Packet Length Variance | 0.6085 | 1.0000 |
| ICMP Code | ICMP Type | 0.4193 | 1.0000 |
| Fwd Bytes/Bulk Avg | Fwd Packet/Bulk Avg | 0.5131 | 1.0000 |
| Fwd Bytes/Bulk Avg | Fwd Bulk Rate Avg | 0.0748 | 1.0000 |
| Fwd Packet/Bulk Avg | Fwd Bulk Rate Avg | 0.0006 | 1.0000 |
| Bwd Bytes/Bulk Avg | Bwd Packet/Bulk Avg | 0.9998 | 0.9996 |
| Flow IAT Max | Idle Max | 0.9995 | 0.7307 |
| Idle Mean | Idle Max | 0.9903 | 0.9995 |
| Fwd Packet Length Min | Packet Length Min | 0.9947 | 0.9992 |
| Flow Packets/s | Fwd Packets/s | 0.9920 | 0.9990 |
| Bwd Bytes/Bulk Avg | Bwd Bulk Rate Avg | 0.0037 | 0.9989 |
| Active Mean | Active Max | 0.9199 | 0.9986 |
| Flow Packets/s | Flow IAT Mean | -0.0283 | -0.9985 |
| Idle Mean | Idle Min | 0.9901 | 0.9984 |
| Bwd Packet/Bulk Avg | Bwd Bulk Rate Avg | 0.0042 | 0.9984 |
| Fwd Packet/Bulk Avg | Fwd Act Data Pkts | 0.9977 | 0.1309 |
| Active Mean | Active Min | 0.9638 | 0.9976 |
| Idle Max | Idle Min | 0.9616 | 0.9972 |
| Flow IAT Mean | Fwd Packets/s | -0.0279 | -0.9972 |
| Total Fwd Packet | Fwd Act Data Pkts | 0.9960 | 0.9571 |
| Total Bwd packets | Total Length of Bwd Packet | 0.9957 | 0.8470 |
| Flow Packets/s | Bwd Packets/s | 0.9916 | 0.9947 |
| Fwd IAT Total | Fwd IAT Max | 0.7446 | 0.9943 |
| Active Max | Active Min | 0.8019 | 0.9942 |
| Flow IAT Mean | Bwd Packets/s | -0.0282 | -0.9940 |
| Flow Duration | Bwd IAT Total | 0.9938 | 0.9246 |
| Total Fwd Packet | Fwd Packet/Bulk Avg | 0.9937 | 0.1311 |
| Fwd PSH Flags | PSH Flag Count | 0.6567 | 0.9932 |
| Bwd IAT Mean | Bwd IAT Max | 0.6175 | 0.9923 |
| Fwd IAT Mean | Fwd IAT Max | 0.7251 | 0.9923 |
| Bwd Segment Size Avg | Subflow Bwd Bytes | 0.9764 | 0.9919 |
| Bwd Packet Length Mean | Subflow Bwd Bytes | 0.9764 | 0.9919 |
| Bwd Packet Length Max | Packet Length Max | 0.6028 | 0.9909 |
| Fwd Packets/s | Bwd Packets/s | 0.9673 | 0.9908 |
| Flow IAT Max | Bwd IAT Max | 0.9905 | 0.9017 |
| Bwd IAT Total | Bwd IAT Max | 0.7715 | 0.9903 |
| Flow IAT Max | Idle Mean | 0.9899 | 0.7303 |
| Bwd IAT Max | Idle Max | 0.9899 | 0.7246 |
| Flow Packets/s | Flow IAT Max | -0.0339 | -0.9892 |
| Flow IAT Max | Fwd Packets/s | -0.0334 | -0.9891 |
| Active Min | Idle Min | 0.4763 | 0.9890 |
| Fwd IAT Total | Fwd IAT Mean | 0.6202 | 0.9889 |
| Bwd IAT Total | Bwd IAT Mean | 0.5460 | 0.9883 |
| Active Mean | Idle Min | 0.4669 | 0.9882 |
| Fwd Segment Size Avg | Subflow Fwd Bytes | 0.9760 | 0.9880 |
| Fwd Packet Length Mean | Subflow Fwd Bytes | 0.9760 | 0.9880 |
| Active Min | Idle Mean | 0.4724 | 0.9868 |
| Active Mean | Idle Max | 0.4583 | 0.9867 |
| Active Mean | Idle Mean | 0.4660 | 0.9867 |
| Active Min | Idle Max | 0.4612 | 0.9864 |
| CWR Flag Count | ECE Flag Count | 0.8994 | 0.9863 |
| Active Max | Idle Min | 0.3960 | 0.9860 |
| Active Max | Idle Max | 0.3959 | 0.9850 |
| Active Max | Idle Mean | 0.3989 | 0.9847 |
| Flow IAT Mean | Flow IAT Max | 0.6246 | 0.9841 |
| Flow Duration | Flow IAT Max | 0.7714 | 0.9839 |
| Total Fwd Packet | Fwd Header Length | 0.0096 | 0.9835 |
| Flow IAT Max | Bwd Packets/s | -0.0338 | -0.9822 |
| Total Bwd packets | ACK Flag Count | 0.9820 | 0.9585 |
| Bwd IAT Max | Idle Mean | 0.9807 | 0.7243 |
| Bwd PSH Flags | PSH Flag Count | 0.8864 | 0.9802 |
| Flow Duration | Fwd Packets/s | -0.0387 | -0.9795 |
| Total Bwd packets | Bwd Header Length | -0.0094 | 0.9792 |
| Active Std | Idle Std | 0.0735 | 0.9788 |
| Total Length of Bwd Packet | ACK Flag Count | 0.9787 | 0.7928 |
| Flow Duration | Flow Packets/s | -0.0392 | -0.9778 |
| Total Length of Bwd Packet | Bwd Packet Length Max | 0.0384 | 0.9770 |
| Packet Length Max | Packet Length Std | 0.9435 | 0.9749 |
| Packet Length Max | Packet Length Variance | 0.6033 | 0.9749 |
| Total Length of Bwd Packet | Packet Length Max | 0.0218 | 0.9725 |
| Flow Duration | Flow IAT Mean | 0.5908 | 0.9706 |
| Bwd Packet Length Max | Bwd Packet Length Std | 0.9703 | 0.8870 |
| Fwd Header Length | ACK Flag Count | 0.1718 | 0.9695 |
| Flow Duration | Bwd Packets/s | -0.0391 | -0.9690 |
| Fwd Packet Length Max | Fwd Packet Length Std | 0.9687 | 0.9196 |
| Bwd Header Length | ACK Flag Count | 0.0028 | 0.9681 |
| Bwd Packet Length Max | Packet Length Std | 0.7063 | 0.9675 |
| Bwd Packet Length Max | Packet Length Variance | 0.0877 | 0.9675 |
| Total Length of Fwd Packet | Fwd Bytes/Bulk Avg | 0.9661 | 0.1190 |
| Fwd Header Length | Bwd Header Length | 0.1601 | 0.9656 |
| Flow IAT Std | Bwd IAT Max | 0.9257 | 0.9642 |
| Bwd Packet Length Mean | Packet Length Mean | 0.8125 | 0.9636 |
| Packet Length Mean | Bwd Segment Size Avg | 0.8125 | 0.9636 |
| Bwd Packet Length Mean | Average Packet Size | 0.8125 | 0.9636 |
| Average Packet Size | Bwd Segment Size Avg | 0.8125 | 0.9636 |
| Flow IAT Std | Bwd IAT Mean | 0.7960 | 0.9630 |
| Total Fwd Packet | ACK Flag Count | 0.0811 | 0.9629 |
| Packet Length Mean | Subflow Bwd Bytes | 0.8152 | 0.9623 |
| Average Packet Size | Subflow Bwd Bytes | 0.8152 | 0.9623 |
| Flow IAT Max | Idle Min | 0.9614 | 0.7286 |
| Total Fwd Packet | Total Bwd packets | 0.0748 | 0.9611 |
| Fwd PSH Flags | Bwd PSH Flags | 0.2329 | 0.9603 |
| Total Length of Fwd Packet | Fwd Packet Length Max | 0.2069 | 0.9598 |
| Protocol | Packet Length Min | 0.8631 | 0.9575 |
| Flow IAT Std | Fwd IAT Total | 0.4977 | 0.9575 |
| Protocol | Fwd Packet Length Min | 0.8545 | 0.9570 |
| Flow IAT Std | Fwd IAT Max | 0.6685 | 0.9560 |
| Flow IAT Std | Bwd IAT Total | 0.7568 | 0.9550 |
| Bwd IAT Max | Idle Min | 0.9523 | 0.7220 |
| Total Bwd packets | Fwd Header Length | 0.1852 | 0.9517 |
| Total Fwd Packet | Bwd Header Length | 0.0019 | 0.9508 |
| Total Length of Bwd Packet | Packet Length Std | 0.0400 | 0.9507 |
| Total Length of Bwd Packet | Packet Length Variance | 0.0080 | 0.9507 |


![2017: Pearson correlation heatmap](results/2_correlation_analysis/cicids2017/pearson_heatmap.png)

![2018: Pearson correlation heatmap](results/2_correlation_analysis/cicids2018/pearson_heatmap.png)

Full per-year flagged pairs: `output/2_correlation_analysis/cicids2017/` and `output/2_correlation_analysis/cicids2018/` (`pearson_flagged_pairs.json`, `spearman_flagged_pairs.json`).

---
## Step 3: Cross-Year Consensus and Feature Removal

A pair flagged in step 2 could be a stable, real redundancy (the same measurement twice) or a one-year coincidence (drift). Step 3 tells these apart by requiring a pair clear BOTH Pearson AND Spearman, in BOTH years (consensus rule: min(|r_2017|, |r_2018|) >= threshold for BOTH Pearson and Spearman, threshold 0.95) before treating it as real redundancy. Found 10 redundant features out of 83 correlation-analysis candidates this way; within each redundancy group, kept the feature with the lowest average correlation to the rest.

Verified against `feature_names.json` directly: every downstream step (training, testing, ablation) runs on 71 final features. Note: 71 is not simply 83−10, since steps 2–3 and step 4 scope their candidate pools slightly differently (step 4 also separately excludes 9 identifier-like columns); both counts are real and correct.

### Redundancy groups (3+ mutually redundant features in both years/metrics)

`Status`: `stable` = both Pearson AND Spearman clear the 0.95 consensus threshold in both years; `stable_pearson_only` = Pearson clears in both years but Spearman falls short (still kept as redundant, since Pearson is the primary signal here, see `3_correlation_comparison.py`'s pair classification). "Avg |r| within group" is the mean absolute correlation across every pair inside the group, averaged across both years.

| Kept | Dropped | Avg Pearson \|r\| | Avg Spearman \|r\| | Status |
|------|---------|------------------:|-------------------:|--------|
| Fwd Packet Length Mean | Fwd Segment Size Avg, Subflow Fwd Bytes | 0.9877 | 0.9895 | stable |
| Bwd Packet Length Mean | Bwd Segment Size Avg, Subflow Bwd Bytes | 0.9893 | 0.9958 | stable |
| Flow Packets/s | Fwd Packets/s | 0.9805 | 0.9990 | stable |
| Fwd URG Flags | URG Flag Count | 0.9998 | 0.9973 | stable |
| Packet Length Mean | Average Packet Size | 1.0000 | 1.0000 | stable |
| Bwd Bytes/Bulk Avg | Bwd Packet/Bulk Avg | 0.9894 | 0.9984 | stable |
| Idle Mean | Idle Max, Idle Min | 0.9893 | 0.9979 | stable |


### Drifted pairs: correlated in one year only, kept (not dropped)

These pairs fail the both-years consensus rule above (one year is far below the threshold), so step 3 keeps both features rather than dropping either; the pair itself is a real cross-year drift signal, separate from the per-feature drift this whole document is otherwise about. 38 such pairs exist, top 10 by |2017-2018 gap| shown:

| Feature A | Feature B | r (2017) | r (2018) | Correlated in |
|---|---|---:|---:|---|
| Total Fwd Packet | Fwd Packet/Bulk Avg | 0.0082 | 0.9937 | 2018_only |
| Total Fwd Packet | Fwd Act Data Pkts | 0.0592 | 0.9960 | 2018_only |
| Total Fwd Packet | Total Bwd packets | 0.9991 | 0.0748 | 2017_only |
| Total Fwd Packet | Total Length of Bwd Packet | 0.9970 | 0.0740 | 2017_only |
| Total Fwd Packet | ACK Flag Count | 0.9997 | 0.0811 | 2017_only |
| Bwd Packet Length Std | Packet Length Variance | 0.9640 | 0.0819 | 2017_only |
| Fwd Packets/s | Bwd Packets/s | 0.2638 | 0.9673 | 2018_only |
| Fwd IAT Min | Bwd IAT Min | 0.9502 | 0.3444 | 2017_only |
| Total Length of Fwd Packet | Fwd Bytes/Bulk Avg | 0.3790 | 0.9661 | 2018_only |
| Flow Packets/s | Bwd Packets/s | 0.4938 | 0.9916 | 2018_only |


To keep in mind: step 2's flag threshold is looser than step 3's drop threshold, by design, since step 2 is exploratory and step 3 is the conservative consensus filter; the step-2 flagged-pairs files list more pairs than step 3 actually drops, and that is expected, not a bug.

The code confirms this directly: in `run_ablation()`, the canonical feature list (`canonical = [f for f in feat17 if f in feat18 ...]`) is sourced from each year's post-drop `feature_names.json`, the same file steps 5 and 6 train and test on. The ablation never sees the dropped features either; this was checked specifically because it is the kind of silent inconsistency that would otherwise undermine the whole H2 test.

Output: `output/3_correlation_comparison/drop_decisions.json`.

![redundancy groups](results/3_correlation_comparison/redundancy_groups.png)

![2018-2017 correlation difference](results/3_correlation_comparison/diff_heatmap.png)

---
## Step 4: Preprocessing

Each year's cleaned data gets its own independent feature set: split into train/test, and a Z-score scaler fit ONLY on that year's training rows. This per-year-only scaler fitting is exactly the mechanism the concept-vs-covariate framing throughout this document depends on.

`feature_names.json`: 71 final features, 10 dropped as redundant (step 3, above), 9 excluded identifier-like columns: `dst ip`, `dst port`, `flow id`, `id`, `label`, `protocol`, `src ip`, `src port`, `timestamp`.

| | 2017 | 2018 |
|---|---:|---:|
| Rows total | 2,099,960 | 63,195,088 |
| Rows train | 1,679,912 | 50,556,009 |
| Rows test | 420,048 | 12,639,079 |
| Test fraction | 0.2000 | 0.2000 |
| Seed | 42 | 42 |

Scaling: a custom streaming Z-score scaler (population variance, ddof=0), saved as two separate artifacts (`output/4_preprocessing/<year>/scaler.json`). Found this difference for two example features, read directly from the saved scaler files rather than hand-typed, to make the magnitude of "own-year scaler" vs "train-year scaler" concrete:

| Feature | 2017 mean | 2017 scale | 2018 mean | 2018 scale |
|---------|----------:|-----------:|----------:|-----------:|
| Flow Duration | 12421417.6723 | 30987847.2035 | 17745041.3262 | 36637831.4737 |
| Down/Up Ratio | 0.9100 | 0.2618 | 0.9445 | 0.3364 |

Fitted on 1679912 training rows (2017) and 50556009 training rows (2018).

### Label mapping (identical for both years)

| Binary label | Code |
|---|---:|
| Benign | 0 |
| Attack | 1 |

| Multiclass family | Code |
|---|---:|
| Benign | 0 |
| Botnet | 1 |
| BruteForce | 2 |
| DDoS | 3 |
| DoS | 4 |
| Infiltration | 5 |
| PortScan | 6 |
| WebAttack | 7 |


![2017: scaling check](results/4_preprocessing/cicids2017/scaling_check.png)

![2018: scaling check](results/4_preprocessing/cicids2018/scaling_check.png)

Output: `output/4_preprocessing/cicids2017/` and `output/4_preprocessing/cicids2018/` contain `feature_names.json`, `preprocessing_meta.json`, `label_mapping.json`, `scaler.json`, and `scaling_check.png`.

---
## Step 5: Model Training and Configuration

Trained one LightGBM random-forest model per year, per task (binary, multiclass), same hyperparameters everywhere, only the data differs.

### Hyperparameters (identical for both years and both tasks)

| Parameter | Value |
|---|---|
| boosting_type | `rf` (bagging, not boosting) |
| n_estimators | `200` |
| num_leaves | `255` |
| max_depth | `-1` (unlimited) |
| bagging_fraction | `0.632` |
| bagging_freq | `1` |
| feature_fraction | `0.5` |
| min_child_samples | `20` |
| class_weight | `'balanced'` |
| random_state | `42` |

### Training summary

| Year | Task | Rows train | Fit time (s) | Model size (MB) | Classes |
|---|---|---:|---:|---:|---:|
| cicids2017 | binary | 1,679,912 | 30.3 | 2.1 | 2 |
| cicids2017 | multiclass | 1,679,912 | 115.8 | 6.9 | 8 |
| cicids2018 | binary | 50,556,009 | 257.8 | 2.2 | 2 |
| cicids2018 | multiclass | 50,556,009 | 1546.0 | 11.4 | 7 |


### Feature Importance

Feature importance is measured three ways:
- **Native GAIN (primary):** total split gain contributed by the feature across all trees, normalized to sum 1 per model. This is the native importance every H1/H2 test reads. Less cardinality-biased than split-count, but still a native tree measure (Strobl 2007 caveat applies).
- **Native split-count (secondary diagnostic):** number of times the feature was used as a split node (normalized to sum 1), saved as `feature_importance_split_<task>.json`; the more cardinality-biased of the two native measures, kept for reference only.
- **Permutation:** balanced-accuracy drop when the feature is randomly shuffled (30 repeats, per-class-capped 25,000-row held-out sample with exact train-duplicates removed first). Unbiased w.r.t. cardinality, but noisy at low importance.

Rows sorted by native (gain) binary 2017 score, highest first. Full importance files: `output/5_training/cicids2017_lightgbm/` and `output/5_training/cicids2018_lightgbm/`.

| Feature | Nat-Bin-17 | Nat-MC-17 | Nat-Bin-18 | Nat-MC-18 | Perm-Bin-17 | Perm-Bin-18 |
|---------|----------:|----------:|----------:|----------:|------------:|------------:|
| RST Flag Count | 0.360402 | 0.001612 | 0.000317 | 0.000574 | 0.005456 | 0.000000 |
| SYN Flag Count | 0.122177 | 0.058430 | 0.000739 | 0.028058 | 0.000179 | 0.000019 |
| Bwd PSH Flags | 0.085552 | 0.019726 | 0.003004 | 0.028833 | 0.000098 | 0.000015 |
| FWD Init Win Bytes | 0.056580 | 0.020297 | 0.390023 | 0.089070 | 0.006050 | 0.092600 |
| PSH Flag Count | 0.042400 | 0.027201 | 0.000892 | 0.029153 | 0.000126 | 0.000015 |
| Flow Duration | 0.029392 | 0.008340 | 0.005753 | 0.025698 | -0.000006 | 0.000008 |
| Fwd Act Data Pkts | 0.029313 | 0.025392 | 0.002936 | 0.005124 | -0.000003 | 0.000005 |
| ACK Flag Count | 0.021134 | 0.002948 | 0.001626 | 0.002673 | 0.000000 | -0.000005 |
| Fwd IAT Min | 0.019578 | 0.007924 | 0.001928 | 0.016372 | 0.000032 | 0.000017 |
| Packet Length Min | 0.018523 | 0.015687 | 0.000026 | 0.000548 | 0.000031 | 0.000108 |
| Flow IAT Max | 0.016052 | 0.013145 | 0.005209 | 0.001577 | -0.000021 | -0.000003 |
| Bwd Packet Length Std | 0.015714 | 0.025631 | 0.027669 | 0.047033 | 0.000164 | 0.000062 |
| Bwd Init Win Bytes | 0.014573 | 0.024115 | 0.223620 | 0.010014 | 0.001113 | 0.000093 |
| Bwd IAT Max | 0.012615 | 0.011720 | 0.005492 | 0.012309 | 0.000007 | 0.000020 |
| Fwd Packet Length Min | 0.012354 | 0.009359 | 0.000301 | 0.001534 | 0.000032 | 0.000109 |
| Fwd Packet Length Max | 0.010814 | 0.023633 | 0.011696 | 0.007696 | 0.000057 | 0.000129 |
| Fwd PSH Flags | 0.010689 | 0.063608 | 0.004371 | 0.040234 | 0.000350 | -0.000006 |
| Fwd Seg Size Min | 0.010462 | 0.009735 | 0.010872 | 0.050774 | 0.000005 | 0.000374 |
| Fwd Packet Length Std | 0.007886 | 0.019573 | 0.010599 | 0.094111 | 0.000005 | 0.000014 |
| Total TCP Flow Time | 0.007819 | 0.027529 | 0.006700 | 0.018122 | -0.000020 | 0.000015 |
| Bwd Packet Length Min | 0.006812 | 0.003469 | 0.000016 | 0.004729 | 0.000010 | 0.000108 |
| Flow Packets/s | 0.006425 | 0.020718 | 0.002838 | 0.002618 | -0.000010 | -0.000004 |
| Packet Length Std | 0.006055 | 0.010516 | 0.003935 | 0.011816 | 0.000053 | 0.000030 |
| Flow IAT Min | 0.005805 | 0.006077 | 0.016504 | 0.002948 | 0.000022 | 0.000032 |
| Total Length of Fwd Packet | 0.005528 | 0.033924 | 0.008773 | 0.011637 | 0.000013 | 0.000012 |
| FIN Flag Count | 0.005479 | 0.001302 | 0.079141 | 0.002738 | 0.000059 | 0.033318 |
| Bwd Packet Length Max | 0.005396 | 0.052927 | 0.037949 | 0.010039 | 0.000000 | 0.000100 |
| Bwd Packet Length Mean | 0.004925 | 0.013346 | 0.009616 | 0.065105 | 0.000035 | 0.000156 |
| Packet Length Variance | 0.004856 | 0.004008 | 0.001247 | 0.019493 | 0.000044 | 0.000001 |
| Total Length of Bwd Packet | 0.004732 | 0.008927 | 0.007500 | 0.004530 | 0.000028 | 0.000046 |
| Total Bwd packets | 0.004532 | 0.002791 | 0.003260 | 0.006043 | 0.000000 | 0.000009 |
| Bwd IAT Mean | 0.003075 | 0.005596 | 0.000915 | 0.004241 | 0.000009 | 0.000019 |
| Bwd IAT Total | 0.002677 | 0.003771 | 0.019787 | 0.004029 | 0.000017 | 0.000013 |
| Subflow Fwd Packets | 0.002597 | 0.000043 | 0.000010 | 0.000164 | 0.000000 | 0.000000 |
| Down/Up Ratio | 0.002555 | 0.059537 | 0.001491 | 0.004544 | 0.000010 | 0.000010 |
| Fwd Packet Length Mean | 0.002432 | 0.010881 | 0.034565 | 0.003480 | 0.000022 | 0.000062 |
| Flow IAT Mean | 0.002432 | 0.043528 | 0.006280 | 0.002539 | -0.000009 | 0.000003 |
| Total Fwd Packet | 0.002330 | 0.054356 | 0.000354 | 0.001985 | 0.000002 | 0.000002 |
| Bwd IAT Min | 0.002102 | 0.038761 | 0.000342 | 0.015101 | 0.000000 | 0.000028 |
| Bwd Packets/s | 0.001606 | 0.011785 | 0.005028 | 0.001745 | 0.000018 | 0.000003 |
| Fwd Header Length | 0.001580 | 0.018656 | 0.003722 | 0.045980 | -0.000000 | 0.000035 |
| Packet Length Mean | 0.001407 | 0.010737 | 0.003896 | 0.048492 | 0.000007 | 0.000038 |
| Packet Length Max | 0.001331 | 0.020972 | 0.022175 | 0.033380 | 0.000010 | 0.000153 |
| Idle Mean | 0.001056 | 0.000645 | 0.000094 | 0.000752 | 0.000001 | 0.000001 |
| Bwd Header Length | 0.000883 | 0.004640 | 0.003032 | 0.015253 | 0.000000 | 0.000018 |
| Flow Bytes/s | 0.000612 | 0.036172 | 0.006637 | 0.101369 | 0.000043 | 0.000035 |
| Bwd IAT Std | 0.000603 | 0.021703 | 0.001085 | 0.004325 | 0.000000 | 0.000024 |
| Fwd IAT Mean | 0.000584 | 0.010616 | 0.000205 | 0.004386 | 0.000002 | 0.000007 |
| Fwd Bytes/Bulk Avg | 0.000540 | 0.002526 | 0.000504 | 0.000517 | 0.000022 | 0.000041 |
| Fwd IAT Std | 0.000489 | 0.000909 | 0.000143 | 0.002399 | 0.000000 | 0.000002 |
| Bwd Bulk Rate Avg | 0.000476 | 0.014075 | 0.000004 | 0.000092 | 0.000008 | 0.000000 |
| Fwd Bulk Rate Avg | 0.000473 | 0.000557 | 0.000095 | 0.000161 | 0.000000 | 0.000016 |
| Fwd IAT Total | 0.000440 | 0.026399 | 0.000164 | 0.011845 | 0.000000 | 0.000013 |
| Fwd RST Flags | 0.000403 | 0.010724 | 0.000077 | 0.002012 | 0.000000 | 0.000006 |
| Active Max | 0.000394 | 0.000497 | 0.000068 | 0.000490 | 0.000000 | 0.000002 |
| Flow IAT Std | 0.000362 | 0.001460 | 0.000308 | 0.000365 | 0.000000 | 0.000011 |
| Fwd Packet/Bulk Avg | 0.000350 | 0.001382 | 0.000593 | 0.000295 | 0.000000 | 0.000005 |
| Fwd IAT Max | 0.000350 | 0.003196 | 0.000175 | 0.005897 | 0.000000 | 0.000031 |
| Bwd RST Flags | 0.000261 | 0.001435 | 0.000676 | 0.000309 | 0.000000 | -0.000000 |
| Active Min | 0.000252 | 0.003580 | 0.001127 | 0.003802 | 0.000000 | 0.000026 |
| Bwd Bytes/Bulk Avg | 0.000232 | 0.004828 | 0.000014 | 0.000194 | 0.000000 | 0.000000 |
| Idle Std | 0.000136 | 0.000751 | 0.000021 | 0.000097 | 0.000000 | 0.000000 |
| Active Std | 0.000111 | 0.000824 | 0.000002 | 0.000781 | 0.000000 | 0.000000 |
| ICMP Type | 0.000109 | 0.000026 | 0.000004 | 0.000057 | -0.000001 | 0.000000 |
| Active Mean | 0.000087 | 0.000773 | 0.000433 | 0.001137 | 0.000000 | 0.000002 |
| ICMP Code | 0.000059 | 0.000034 | 0.000037 | 0.000256 | 0.000011 | 0.000264 |
| Fwd URG Flags | 0.000037 | 0.000010 | 0.000098 | 0.000041 | 0.000000 | -0.000000 |
| ECE Flag Count | 0.000004 | 0.000000 | 0.000513 | 0.012284 | 0.000000 | 0.000380 |
| CWR Flag Count | 0.000002 | 0.000001 | 0.000772 | 0.014002 | 0.000000 | 0.000199 |
| Bwd URG Flags | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Subflow Bwd Packets | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

> Near-zero permutation values are expected: tree ensembles use feature combinations, so single-feature shuffling understates individual contributions.

![2017: native importance, binary](results/5_training/cicids2017_lightgbm/feature_importance_binary.png)

![2017: native importance, multiclass](results/5_training/cicids2017_lightgbm/feature_importance_multiclass.png)

![2017: permutation importance, binary](results/5_training/cicids2017_lightgbm/feature_importance_perm_binary.png)

![2017: permutation importance, multiclass](results/5_training/cicids2017_lightgbm/feature_importance_perm_multiclass.png)

![2018: native importance, binary](results/5_training/cicids2018_lightgbm/feature_importance_binary.png)

![2018: native importance, multiclass](results/5_training/cicids2018_lightgbm/feature_importance_multiclass.png)

![2018: permutation importance, binary](results/5_training/cicids2018_lightgbm/feature_importance_perm_binary.png)

![2018: permutation importance, multiclass](results/5_training/cicids2018_lightgbm/feature_importance_perm_multiclass.png)

Output: `output/5_training/cicids2017_lightgbm/` and `output/5_training/cicids2018_lightgbm/` contain feature importance CSVs and training metadata.

---
## Step 6: Cross-Year Test Results

Each trained model is evaluated in two framings against the opposite year's data:
- **Concept:** target year normalized with its *own* scaler, covariate shift removed, tests decision-boundary transfer.
- **Covariate:** target year normalized with the *train-year* scaler, the deployment reality where only the old scaler exists.

> ⚠️ Within-year scores (17→17, 18→18) use a per-row hash split that allows near-duplicate flows to straddle train/test. They are inflated baselines, not a gold standard.

### Binary (benign vs attack)

_Full metric table, same-year baseline (own 20% held-out split) alongside all four cross-year cells, with the gap from same-year made explicit, one table, not split across a metric-breakdown table and a separate gap table that repeated the same cross-year numbers. Same metric set as C9's ablation table below (this is the real full-feature model, the ceiling the K-feature ablation policies are compared against). Same-year rows use their own within-year 20% test split (⚠️ inflated by near-duplicate train/test flows, see the warning above), not a fifth/sixth cross framing, there is no "own-scaler, own-year" cross case to add, since concept and covariate framings only differ when train-year != test-year. Gap columns = same-year baseline minus that row's value; POSITIVE means performance LOST moving out of domain (e.g. a drop from 0.9999 to 0.9211 prints as 0.0788, not -0.0788).

| Cell | Accuracy | Macro F1 | Attack F1 | Benign F1 | Sensitivity | FPR | Precision | Specificity | Balanced Acc | ROC-AUC | MCC | Acc gap (same−cross) | Macro-F1 gap (same−cross) |
|------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Same-year 2017 (baseline) | 0.9999 | 0.9998 | 0.9997 | 0.9999 | 0.9997 | 0.0001 | 0.9997 | 0.9999 | 0.9998 | 1.0000 | 0.9997 | (baseline) | (baseline) |
| 2017→2018 concept | 0.9211 | 0.7564 | 0.5561 | 0.9567 | 0.8834 | 0.0767 | 0.4058 | 0.9233 | 0.9034 | 0.9337 | 0.5669 | 0.0788 | 0.2434 |
| 2017→2018 covariate | 0.9393 | 0.5062 | 0.0437 | 0.9686 | 0.0248 | 0.0065 | 0.1839 | 0.9935 | 0.5091 | 0.8289 | 0.0486 | 0.0606 | 0.4936 |
| Same-year 2018 (baseline) | 1.0000 | 0.9998 | 0.9996 | 1.0000 | 0.9997 | 0.0000 | 0.9995 | 1.0000 | 0.9998 | 1.0000 | 0.9995 | (baseline) | (baseline) |
| 2018→2017 concept | 0.7593 | 0.4317 | 0.0001 | 0.8632 | 0.0000 | 0.0000 | 0.8333 | 1.0000 | 0.5000 | 0.8767 | 0.0052 | 0.2406 | 0.5681 |
| 2018→2017 covariate | 0.8164 | 0.6424 | 0.3930 | 0.8918 | 0.2469 | 0.0031 | 0.9616 | 0.9969 | 0.6219 | 0.9187 | 0.4328 | 0.1836 | 0.3574 |

> Covariate 17→18 attack F1 = 0.0437, the "faint detection" that motivated this study. Concept framing recovers 17→18 partially (attack F1 = 0.5561); 18→17 collapses in both framings (concept attack F1 = 0.0001).

Found a large gap in all four cross-year cells, which means the same-year ~0.9999 accuracy/macro-F1 numbers are not a usable estimate of deployed performance for either year. To keep in mind: the same-year numbers are themselves inflated by the near-duplicate-flow leakage noted above, so this gap is a lower bound on the real drop, not the full size of it.

### Multiclass (8 attack families)

| Metric | 17→18 concept | 17→18 covariate | 18→17 concept | 18→17 covariate |
|--------|------:|------:|------:|------:|
| Macro F1    | 0.1852 | 0.2445 | 0.1079 | 0.2557 |
| Balanced Acc| 0.2643 | 0.3204 | 0.1250 | 0.2664 |
| MCC         | 0.0067 | 0.0397 | 0.0000 | 0.2268 |

### Supplementary figures: ROC, PR, and confusion matrices

![2017 same-year: binary ROC](results/6_testing/cicids2017_lightgbm/6_testing_roc_binary.png)

![2017 same-year: binary Precision-Recall](results/6_testing/cicids2017_lightgbm/6_testing_pr_binary.png)

![2017 same-year: binary confusion matrix](results/6_testing/cicids2017_lightgbm/6_testing_confusion_binary.png)

![2017 same-year: multiclass confusion matrix](results/6_testing/cicids2017_lightgbm/6_testing_confusion_multiclass.png)

![2017 same-year: multiclass per-class F1/recall](results/6_testing/cicids2017_lightgbm/6_testing_per_class_f1_multiclass.png)

![2018 same-year: binary ROC](results/6_testing/cicids2018_lightgbm/6_testing_roc_binary.png)

![2018 same-year: binary Precision-Recall](results/6_testing/cicids2018_lightgbm/6_testing_pr_binary.png)

![2018 same-year: binary confusion matrix](results/6_testing/cicids2018_lightgbm/6_testing_confusion_binary.png)

![2018 same-year: multiclass confusion matrix](results/6_testing/cicids2018_lightgbm/6_testing_confusion_multiclass.png)

![2018 same-year: multiclass per-class F1/recall](results/6_testing/cicids2018_lightgbm/6_testing_per_class_f1_multiclass.png)

![2017->2018 [concept]: binary ROC](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_roc_binary_concept.png)

![2017->2018 [concept]: binary Precision-Recall](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_pr_binary_concept.png)

![2017->2018 [concept]: binary confusion matrix](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_confusion_binary_concept.png)

![2017->2018 [concept]: multiclass confusion matrix](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_confusion_multiclass_concept.png)

![2017->2018 [covariate]: binary ROC](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_roc_binary_covariate.png)

![2017->2018 [covariate]: binary Precision-Recall](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_pr_binary_covariate.png)

![2017->2018 [covariate]: binary confusion matrix](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_confusion_binary_covariate.png)

![2017->2018 [covariate]: multiclass confusion matrix](results/6_testing/cross_cicids2017_to_cicids2018_lightgbm/6_cross_confusion_multiclass_covariate.png)

![2018->2017 [concept]: binary ROC](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_roc_binary_concept.png)

![2018->2017 [concept]: binary Precision-Recall](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_pr_binary_concept.png)

![2018->2017 [concept]: binary confusion matrix](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_confusion_binary_concept.png)

![2018->2017 [concept]: multiclass confusion matrix](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_confusion_multiclass_concept.png)

![2018->2017 [covariate]: binary ROC](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_roc_binary_covariate.png)

![2018->2017 [covariate]: binary Precision-Recall](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_pr_binary_covariate.png)

![2018->2017 [covariate]: binary confusion matrix](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_confusion_binary_covariate.png)

![2018->2017 [covariate]: multiclass confusion matrix](results/6_testing/cross_cicids2018_to_cicids2017_lightgbm/6_cross_confusion_multiclass_covariate.png)

---
## Step 7: Feature Profiles

Step 7 computed distributional statistics for each of the 71 features in both 2017 and 2018 separately. Computed properties include: detected distribution type (nominal / continuous / discrete_count), zero-inflation flag, number of detected modes, full percentile table (p1–p99), skewness, kurtosis, entropy, mutual information with the attack/benign label (MI), and separation AUC (univariate ability to distinguish benign from attack traffic with that feature alone).

Full profiles: `output/7_profile/cicids2017/profiles.json` and `output/7_profile/cicids2018/profiles.json`.

### Cardinality (detected_type breakdown)

Step 7 does not store a raw per-feature unique-value count, only a categorical `detected_type` bucket; this is the cardinality-adjacent breakdown that is actually available, counted across all features profiled in each year (note: step 7 profiled all 83 pre-drop features, not the 71-feature post-drop set used for training).

| detected_type | 2017 count | 2018 count |
|---------------|-----------:|-----------:|
| binary | 3 | 3 |
| continuous | 65 | 67 |
| discrete-count | 6 | 7 |
| low-cardinality-discrete | 7 | 4 |
| nominal | 2 | 2 |


### Worked example: Total Length of Bwd Packet (2017)

Every field from the 2017 profile JSON for this one feature, grouped by category (the other 70 features have the same fields, see the JSON pointer above rather than repeating this block 70 more times):

```
# identity / type flags
detected_type            : continuous
zero_inflated            : True
has_sentinel             : False
sentinel_value           : n/a
has_impossible_values    : False
degenerate               : False
is_identifier            : False

# counts / missingness
n_total                  : 2099960
n_missing                : 0
n_zero                   : 331909
zero_fraction            : 0.1581

# location / spread
min / max                : 0.0000 / 655452323.0000
mean / median / std      : 21881.8700 / 218.0000 / 2627413.1156
range / iqr / mad        : 655452323.0000 / 3781.0000 / 218.0000
coefficient_of_variation : 120.0726

# percentiles
p01 / p05 / p10          : 0.0000 / 0.0000 / 0.0000
p25 / p50 / p75          : 100.0000 / 218.0000 / 3881.0000
p90 / p95 / p99          : 11595.0000 / 11595.0000 / 101535.0000

# shape
skewness                 : 210.4573
kurtosis                 : 45702.9057
entropy                  : 0.0013

# outliers / sentinel mass
outlier_count_low/high   : 0 / 355601
outlier_fraction         : 0.1693
sentinel_mass            : 0.0000

# modes (multi-modality detection)
n_modes                  : 3
modes                    : (center=181.00, spread=189.13, mass=0.64), (center=11595.00, spread=5759.44, mass=0.33), (center=94796.50, spread=26439677.20, mass=0.03)

# recommended scaling
recommended_scale        : symlog
scale_param              : 97.0

# clipping (view window used for plots, not the actual data)
view_low / view_high     : 0.0000 / 203552.7150
n_clipped_low/high       : 0 / 10500

# attack/benign separation (univariate, this feature alone)
roc_auc_benign_vs_attack : 0.5162
separation_magnitude     : 0.5162
separation_direction     : 0.0  (1 = attack values higher, -1 = benign values higher)
separation_auc_raw       : 0.4838
mutual_info              : 0.4241
mutual_info_normalized   : 0.8628
```

`per_class_separation`, the same separation test repeated per attack family (benign vs that one family only):
```
  DoS            magnitude=0.9151  direction=1.0  mi_normalized=0.9555  n_attack=60000
  Botnet         magnitude=0.7121  direction=-1.0  mi_normalized=0.8549  n_attack=736
  PortScan       magnitude=0.9687  direction=-1.0  mi_normalized=0.8273  n_attack=60000
  DDoS           magnitude=0.9435  direction=1.0  mi_normalized=0.9970  n_attack=60000
  WebAttack      magnitude=0.9586  direction=1.0  mi_normalized=0.8149  n_attack=104
  Infiltration   magnitude=0.9652  direction=-1.0  mi_normalized=0.8167  n_attack=60000
  BruteForce     magnitude=0.5895  direction=1.0  mi_normalized=0.9217  n_attack=6933
```

> `mutual_info_normalized` and `roc_auc_benign_vs_attack` from both years feed directly into the Axis 2 (concept stability) calculation in Step 10. `detected_type` and mode count drive the Step 9 routing decision. `zero_fraction` is reported again below alongside the Axis 1 shift metrics (C2ST/MMD/Wasserstein) as data-quality context, not a drift measurement.

### MI and separation AUC: all 71 features, both years

Sorted by MI_2017 descending. AUC below 0.55 indicates near-chance separation.

| Feature | MI_2017 | AUC_2017 | MI_2018 | AUC_2018 |
|---------|--------:|---------:|--------:|---------:|
| Bwd Packet Length Mean | 0.8738 | 0.5030 | 0.8668 | 0.5656 |
| Total Length of Bwd Packet | 0.8628 | 0.5162 | 0.8619 | 0.5291 |
| Packet Length Max | 0.8504 | 0.5036 | 0.8840 | 0.5332 |
| Packet Length Mean | 0.8382 | 0.5071 | 0.8046 | 0.5653 |
| Bwd Packet Length Max | 0.8373 | 0.5043 | 0.8558 | 0.5051 |
| Packet Length Std | 0.8291 | 0.5150 | 0.8422 | 0.5553 |
| Packet Length Variance | 0.8290 | 0.5150 | 0.8421 | 0.5554 |
| Bwd Header Length | 0.7974 | 0.6256 | 0.7618 | 0.6658 |
| Fwd Packet Length Max | 0.7858 | 0.7724 | 0.8311 | 0.5581 |
| Total TCP Flow Time | 0.7715 | 0.6861 | 0.7413 | 0.5073 |
| Fwd Header Length | 0.7613 | 0.6878 | 0.7716 | 0.6476 |
| Total Length of Fwd Packet | 0.7446 | 0.7740 | 0.7983 | 0.5478 |
| Flow Duration | 0.6678 | 0.6431 | 0.4851 | 0.6075 |
| Flow IAT Max | 0.6426 | 0.6410 | 0.4357 | 0.6208 |
| Fwd Packet Length Mean | 0.6310 | 0.8150 | 0.7407 | 0.5435 |
| FWD Init Win Bytes | 0.6018 | 0.7677 | 0.5672 | 0.8287 |
| Fwd Seg Size Min | 0.5778 | 0.8406 | 0.4550 | 0.8646 |
| SYN Flag Count | 0.5560 | 0.7825 | 0.3570 | 0.7016 |
| ACK Flag Count | 0.5226 | 0.6542 | 0.7094 | 0.6266 |
| Bwd Packets/s | 0.4933 | 0.5711 | 0.3667 | 0.6576 |
| Flow Packets/s | 0.4909 | 0.6734 | 0.3647 | 0.6586 |
| Fwd Packet Length Min | 0.4879 | 0.8127 | 0.3042 | 0.7075 |
| Packet Length Min | 0.4869 | 0.8127 | 0.3040 | 0.7074 |
| Bwd Packet Length Min | 0.4841 | 0.8101 | 0.3151 | 0.7115 |
| RST Flag Count | 0.4646 | 0.8787 | 0.0269 | 0.5695 |
| Flow IAT Mean | 0.4634 | 0.6707 | 0.3672 | 0.6844 |
| Flow Bytes/s | 0.4602 | 0.7184 | 0.4154 | 0.5249 |
| Bwd Packet Length Std | 0.3993 | 0.6606 | 0.6404 | 0.6027 |
| Bwd IAT Max | 0.3986 | 0.5748 | 0.4658 | 0.5318 |
| Bwd IAT Total | 0.3948 | 0.5790 | 0.4699 | 0.5308 |
| Bwd Init Win Bytes | 0.3863 | 0.5667 | 0.6048 | 0.6576 |
| Bwd IAT Mean | 0.3576 | 0.5742 | 0.4660 | 0.5425 |
| Flow IAT Std | 0.3404 | 0.5899 | 0.4013 | 0.5762 |
| Total Bwd packets | 0.3296 | 0.5519 | 0.4910 | 0.5886 |
| Fwd Packet Length Std | 0.3019 | 0.5441 | 0.5886 | 0.6232 |
| Fwd IAT Max | 0.2946 | 0.5178 | 0.4022 | 0.5643 |
| Fwd IAT Total | 0.2820 | 0.5241 | 0.3902 | 0.5547 |
| Total Fwd Packet | 0.2457 | 0.5182 | 0.4471 | 0.5791 |
| PSH Flag Count | 0.2421 | 0.5369 | 0.4727 | 0.5852 |
| Bwd IAT Min | 0.2338 | 0.5480 | 0.3504 | 0.5230 |
| Bwd PSH Flags | 0.2338 | 0.5387 | 0.4356 | 0.5976 |
| Fwd IAT Mean | 0.2228 | 0.5248 | 0.3843 | 0.5864 |
| Down/Up Ratio | 0.2224 | 0.6519 | 0.2343 | 0.5520 |
| Bwd IAT Std | 0.2172 | 0.5349 | 0.4613 | 0.5166 |
| Flow IAT Min | 0.2170 | 0.5669 | 0.4187 | 0.7076 |
| Fwd PSH Flags | 0.2134 | 0.5392 | 0.4642 | 0.5695 |
| Fwd Act Data Pkts | 0.1637 | 0.6929 | 0.3965 | 0.5439 |
| Fwd IAT Std | 0.1425 | 0.5289 | 0.4219 | 0.5258 |
| Bwd Bytes/Bulk Avg | 0.1355 | 0.5551 | 0.0484 | 0.5327 |
| Fwd RST Flags | 0.1338 | 0.6993 | 0.0534 | 0.5864 |
| Bwd RST Flags | 0.1266 | 0.6810 | 0.0087 | 0.5124 |
| Fwd IAT Min | 0.1224 | 0.6577 | 0.2937 | 0.6389 |
| FIN Flag Count | 0.1198 | 0.5491 | 0.2525 | 0.7929 |
| Active Min | 0.1094 | 0.5342 | 0.1468 | 0.5868 |
| Bwd Bulk Rate Avg | 0.1093 | 0.5634 | 0.0485 | 0.5327 |
| Idle Mean | 0.1051 | 0.5462 | 0.1279 | 0.5910 |
| Active Std | 0.0771 | 0.5590 | 0.0783 | 0.5593 |
| Idle Std | 0.0747 | 0.5600 | 0.0817 | 0.5570 |
| Active Mean | 0.0724 | 0.5364 | 0.1386 | 0.5871 |
| Active Max | 0.0670 | 0.5409 | 0.1334 | 0.5878 |
| Fwd Bulk Rate Avg | 0.0265 | 0.5215 | 0.0089 | 0.5011 |
| Fwd Bytes/Bulk Avg | 0.0265 | 0.5211 | 0.0073 | 0.5011 |
| Subflow Fwd Packets | 0.0201 | 0.5452 | 0.0034 | 0.5027 |
| Fwd Packet/Bulk Avg | 0.0164 | 0.5212 | 0.0050 | 0.5011 |
| ICMP Type | 0.0159 | 0.5000 | 0.0390 | 0.5007 |
| ICMP Code | 0.0155 | 0.5000 | 0.0368 | 0.5007 |
| Bwd URG Flags | 0.0000 | 0.5000 | 0.0037 | 0.5000 |
| CWR Flag Count | 0.0000 | 0.5002 | 0.0207 | 0.5619 |
| ECE Flag Count | 0.0000 | 0.5001 | 0.0880 | 0.5389 |
| Fwd URG Flags | 0.0000 | 0.5002 | 0.0046 | 0.5009 |
| Subflow Bwd Packets | 0.0000 | 0.5000 | 0.0043 | 0.5004 |

### Supporting-analyses index

A short map from each supporting test to where it lives in this document, for anyone looking for a specific diagnostic:

| Supporting analysis | Where it is |
|---|---|
| Cardinality | New table just above (`detected_type` breakdown); no raw unique-count field exists upstream |
| Variance | Not broken out as its own table; not informative on its own for this feature set |
| Per-class separation | Step 7 worked example below, and C2b's Table B (feature x family matrix) |
| Sensitivity sweep (robustness check, verdict counts at 3 alternate calibrated-C2ST verdict thresholds; no C-number, since it is not rendered in this document, see the "where" column) | Not in results.md; full table in Step 10's own text report, `results/10_execute_comparison/10_execute_comparison_report.txt` |
| Metric agreement (Wasserstein/MMD/KS/Energy/Anderson-Darling vs C2ST-AUC) | See E1, Supplementary checks (E-series) at the end of Step 11 |
| Cluster bootstrap | In the H1 headline cells (C8): each cell's 95% CI and preferred p-value are cluster-bootstrap (resampling collinearity clusters) where available |
| Partial correlation | Not broken out as its own table; the cluster-bootstrap CI already accounts for collinearity |

---
## Step 8: (skipped)

Step 8 is visualization-only (renders Step 7's profiles to PNGs for manual inspection) and produces no numbers consumed downstream; the pipeline goes straight from Step 7's profiles to Step 9's routing decision.

---
## Step 9: Comparison Planning (Routing)

Step 9 reads the Step 7 distribution profiles and assigns each feature a comparison route, the CORROBORATION-metric family appropriate for its type. The route only picks which corroboration distances step 10 computes (and whether comparison runs per mode or on the whole distribution); the stable/shifted VERDICT is always decided by calibrated C2ST-AUC, the one metric computed identically for every route.

| Route | Corroboration Primary | Count | Comparison Mode |
|-------|---------------|------:|----------------|
| `continuous_multimodal` | mmd | 39 | per_mode |
| `structural_change` | wasserstein_qn | 31 | whole_distribution |
| `nominal` | jensen_shannon | 9 | whole_distribution |
| `discrete_count` | jensen_shannon | 4 | whole_distribution |

Route conditions (verdict metric is always calibrated C2ST-AUC; the metrics named below are the corroboration battery only):
- `structural_change`: type or modality changed 2017↔2018; whole-distribution corroboration distances.
- `continuous_multimodal`: multiple modes detected; per-mode MMD corroboration, then aggregates.
- `nominal`: categorical or flag feature; Jensen-Shannon divergence on the PMF.
- `discrete_count`: low-cardinality integer count; Jensen-Shannon divergence on the full PMF.

### How each metric is calculated

Step 9 only assigns which of these a feature uses; step 10 (next) runs the actual calculation and is where the numeric results live.

| Metric | Method |
|---|---|
| C2ST-AUC | Classifier Two-Sample Test (Lopez-Paz and Oquab 2017): train a shallow decision tree (max_depth=6) to predict whether a row is from 2017 or 2018 using this one feature alone, with 5-fold stratified cross-validation; report the mean held-out AUC plus a CI built from the per-fold scores. 0.5 = indistinguishable years, 1.0 = perfectly separable. |
| MMD | Unbiased squared Maximum Mean Discrepancy with an RBF kernel, median-heuristic bandwidth (Gretton et al. 2012): a kernel-based distance between the two years' empirical distributions. 0 = identical, larger = more different; sensitive to any kind of distributional change (location, scale, or shape). |
| Wasserstein-qn | Wasserstein (earth-mover) distance computed AFTER rank-normalizing both years onto a common reference distribution, which strips out location and scale first, so this isolates shape-only differences. |
| Jensen-Shannon divergence | Symmetric, bounded divergence between the two years' probability mass functions (top-50 categories plus an "other" bucket for high-cardinality features); used instead of a continuous-distance metric for nominal/discrete-count features, since categories have no natural ordering or distance between them. |

Full routing plan: `output/9_plan_comparison/comparison_plans_cicids2017_cicids2018.json`.

---
## Step 10: Drift Axes

Step 10 executes the statistical comparisons planned in Step 9 and produces two axes per feature:

> **Quick terms** (so "covariate," "covariance," and "cardinality" don't blur together below, skip this box if they already don't): **Covariate shift** (Axis 1) = did the feature's VALUES move between years. Not "covariance", covariance is a different statistic (how two variables move together) and is never used as an axis name anywhere in this document. **Concept stability** (Axis 2) = did the feature's RELATIONSHIP to the attack/benign label survive. **Cardinality** = how many distinct values a feature takes; unrelated to either axis, it matters only because native tree importance (gain, and split-count even more so) is biased toward high-cardinality features (Strobl 2007), which is why several tests in Section 2 below control for it.

**Axis 1: Covariate shift (`cov_shift`)**
Measures whether the feature's value distribution moved between 2017 and 2018. Primary metric: C2ST-AUC (train a classifier to distinguish 2017 from 2018 values; 0.5 = indistinguishable, 1.0 = fully shifted), CALIBRATED against a per-feature, per-slice NULL FLOOR: pool both years, randomly re-split into two same-size halves many times, recompute C2ST-AUC each time, the floor is the AUC you would still see even if the two years were identical. Calibrated = (raw − null floor) / (1 − null floor), clipped to [0, 1]. The 5-level Axis-1 status is bucketed off this CALIBRATED value (0 = at/below its own null floor, not the fixed 0.5 chance level):
  NONE ≤ 0 | LOW 0–0.20 | MODERATE 0.20–0.45 | HIGH 0.45–0.70 | STRONG > 0.70
The stable/shifted `verdict` column is decided by this SAME calibrated C2ST value (> 0 = shifted), one Axis-1 decision rule everywhere. The corroboration distances (Wasserstein-qn / MMD / energy / KS / Anderson-Darling, or Jensen-Shannon for PMF-routed features) are each calibrated against their OWN permutation null, POOLED ONLY, and used purely for the E1 agreement check, they never decide anything. C2ST is additionally computed for benign-only, attack-only, and each individual attack family, each slice gets its OWN null floor (a smaller slice is noisier, so its floor is naturally higher; a raw value from a noisy slice is not directly comparable to a raw value from a clean one, which is exactly what this per-slice calibration fixes). `marginal_shift` remains in the tables as a DESCRIPTIVE route-family distance only.

**Axis 2: Concept stability (`concept_stab` / separation_stability)**
Measures whether the feature still separates attack from benign in 2018. Per year, separation strength is `max((AUC − 0.5) × 2, MI_normalized)` (MI-aware: a feature whose separation is non-monotonic, e.g. multimodal, still counts if normalized mutual information detects it even though folded-AUC alone would not). The two years' strengths are clipped to [0, 1] and multiplied; the product is negated if both years are strongly separated (clears the null-calibrated AUC/MI floor) with opposite TRUSTED directions, i.e. a genuine flip. `separation_stability_auc`, a legacy folded-AUC-only version of the same formula (no MI term), is also saved in Layer B and is what the per-attack-family / per-class breakdown table below uses, since per-family MI is not computed separately. PRESERVED ≥ 0.35 | WEAKENED 0.09–0.35 | COLLAPSED 0–0.09 | FLIPPED < 0.

The full Axis 1 table (with null floors and per-attack-family C2ST) and full Axis 2 table (per-attack-family separation stability) are in the root `results.md`. The table below shows the pooled summary from `cross_table.csv`.

### Axis 1 + 2 summary: all 71 features (sorted by calibrated C2ST descending)

`Quadrant` crosses 2017 native importance with concept stability (Axis 2) into a 2x2 grid:

| Quadrant | Meaning |
|---|---|
| `Q1_good` | high-importance, stable: model relies on transferable features |
| `Q2_fragile_shortcut` | high-importance, unstable: model relies on features that break |
| `Q3_noise` | low-importance, unstable |
| `Q4_underused_stable` | low-importance, stable: a reservoir of stable-but-unused features |

`Verdict` is a priority cascade over Axis 1 and Axis 2 together (first match wins; the full rule with its exact conditions is spelled out later at "How the `verdict` column is actually decided"); the short version:

| Verdict | Meaning |
|---|---|
| `flipped` | separation was trusted in both years, but the attack/benign direction reversed |
| `collapsed` | separation was trusted in 2017 but lost by 2018 |
| `weak` | separation was never trusted in either year |
| `restructured` | the feature's distribution shape (modality/type) changed between years AND calibrated C2ST confirms the mismatch is real, not GMM/type-detection noise |
| `shifted` | calibrated C2ST-AUC is above threshold and none of the above apply |
| `stable` | none of the above; calibrated C2ST-AUC at or below threshold |

_This is the full per-feature detail table, Axis 1 (C2ST pooled + the benign-only/attack-only slice C2ST, both raw and calibrated, slices carry C2ST only; corroboration distances are pooled-only by design) and Axis 2 (separation stability), plus verdict and quadrant. C2a further below shows the same slice C2ST next to each slice's null floor._

| Feature | C2ST pooled (raw) | C2ST pooled (calibrated) | Benign C2ST (raw) | Benign C2ST (calibrated) | Attack C2ST (raw) | Attack C2ST (calibrated) | Sep-Stab | Verdict | Quadrant |
|---------|-----:|-----:|-----:|-----:|-----:|-----:|---------:|---------|---------|
| Bwd IAT Min | 0.8502 | 0.6857 | 0.8740 | 0.7380 | 0.9508 | 0.8959 | 0.0819 | shifted | Q3_noise |
| Flow IAT Max | 0.8279 | 0.6376 | 0.8413 | 0.6764 | 0.9220 | 0.8381 | 0.2800 | shifted | Q4_underused_stable |
| Flow Duration | 0.8228 | 0.6345 | 0.8145 | 0.6032 | 0.9143 | 0.8239 | 0.3239 | restructured | Q4_underused_stable |
| Bwd Packets/s | 0.8222 | 0.6260 | 0.8173 | 0.6202 | 0.8719 | 0.7334 | 0.1809 | restructured | Q1_good |
| Flow Packets/s | 0.8147 | 0.6147 | 0.8291 | 0.6368 | 0.8855 | 0.7595 | 0.1790 | restructured | Q4_underused_stable |
| Flow IAT Min | 0.8066 | 0.5995 | 0.8471 | 0.6838 | 0.9154 | 0.8240 | 0.0909 | shifted | Q2_fragile_shortcut |
| Fwd IAT Mean | 0.7991 | 0.5833 | 0.8097 | 0.6095 | 0.8676 | 0.7281 | 0.0856 | shifted | Q2_fragile_shortcut |
| Fwd IAT Min | 0.7981 | 0.5811 | 0.7564 | 0.4956 | 0.9630 | 0.9227 | -0.0926 | flipped | Q2_fragile_shortcut |
| Fwd IAT Max | 0.7923 | 0.5768 | 0.8188 | 0.6175 | 0.9093 | 0.8106 | 0.1185 | shifted | Q4_underused_stable |
| Flow IAT Mean | 0.7904 | 0.5703 | 0.8034 | 0.5961 | 0.8811 | 0.7597 | 0.1708 | restructured | Q4_underused_stable |
| Total Length of Fwd Packet | 0.7906 | 0.5533 | 0.8251 | 0.6326 | 0.8418 | 0.6637 | 0.5944 | shifted | Q1_good |
| Packet Length Variance | 0.7850 | 0.5492 | 0.7197 | 0.4263 | 0.9909 | 0.9813 | 0.6981 | shifted | Q1_good |
| Packet Length Max | 0.7848 | 0.5483 | 0.7114 | 0.3971 | 0.9951 | 0.9900 | 0.7518 | restructured | Q1_good |
| Bwd Header Length | 0.7834 | 0.5460 | 0.7862 | 0.5524 | 0.9393 | 0.8683 | 0.6074 | restructured | Q4_underused_stable |
| Bwd Packet Length Max | 0.7832 | 0.5435 | 0.7487 | 0.4701 | 0.9915 | 0.9823 | 0.7165 | restructured | Q4_underused_stable |
| Bwd IAT Max | 0.7768 | 0.5428 | 0.8229 | 0.6336 | 0.8804 | 0.7483 | 0.1857 | shifted | Q1_good |
| Total Length of Bwd Packet | 0.7772 | 0.5398 | 0.7682 | 0.5202 | 0.9923 | 0.9836 | 0.7436 | restructured | Q1_good |
| Fwd IAT Total | 0.7780 | 0.5383 | 0.7968 | 0.5776 | 0.8894 | 0.7653 | 0.1100 | shifted | Q3_noise |
| Fwd Header Length | 0.7800 | 0.5315 | 0.7736 | 0.5336 | 0.9819 | 0.9617 | 0.5874 | shifted | Q4_underused_stable |
| Packet Length Std | 0.7679 | 0.5188 | 0.7223 | 0.4197 | 0.9916 | 0.9822 | 0.6982 | shifted | Q1_good |
| Bwd IAT Mean | 0.7560 | 0.4907 | 0.8225 | 0.6339 | 0.8175 | 0.6220 | 0.1666 | restructured | Q1_good |
| FWD Init Win Bytes | 0.7558 | 0.4863 | 0.7337 | 0.4375 | 0.9787 | 0.9547 | 0.3956 | shifted | Q1_good |
| Bwd Init Win Bytes | 0.7365 | 0.4602 | 0.6912 | 0.3525 | 0.9923 | 0.9833 | 0.2337 | restructured | Q1_good |
| Bwd Packet Length Std | 0.7308 | 0.4368 | 0.6606 | 0.2993 | 0.9901 | 0.9792 | 0.2557 | shifted | Q1_good |
| Flow Bytes/s | 0.7319 | 0.4308 | 0.7954 | 0.5749 | 0.9029 | 0.7967 | 0.1912 | restructured | Q1_good |
| Total Bwd packets | 0.7243 | 0.4235 | 0.7470 | 0.4758 | 0.9293 | 0.8485 | -0.1618 | flipped | Q3_noise |
| Bwd IAT Total | 0.7224 | 0.4190 | 0.8005 | 0.5888 | 0.8377 | 0.6625 | 0.1855 | shifted | Q1_good |
| Fwd Packet Length Max | 0.7243 | 0.4167 | 0.7221 | 0.4255 | 0.8368 | 0.6590 | -0.6531 | flipped | Q2_fragile_shortcut |
| Packet Length Mean | 0.7169 | 0.4011 | 0.6745 | 0.3029 | 0.9922 | 0.9838 | 0.6744 | restructured | Q1_good |
| Flow IAT Std | 0.7073 | 0.3960 | 0.7588 | 0.4982 | 0.9079 | 0.8029 | 0.1366 | restructured | Q4_underused_stable |
| Total Fwd Packet | 0.7059 | 0.3950 | 0.7408 | 0.4508 | 0.9779 | 0.9537 | 0.1099 | shifted | Q2_fragile_shortcut |
| Fwd Packet Length Std | 0.6927 | 0.3715 | 0.6680 | 0.3103 | 0.9121 | 0.8157 | 0.1777 | shifted | Q1_good |
| Fwd Seg Size Min | 0.7001 | 0.3705 | 0.6942 | 0.3679 | 0.7909 | 0.5704 | 0.4967 | restructured | Q1_good |
| Bwd Packet Length Mean | 0.6969 | 0.3681 | 0.6667 | 0.3054 | 0.9922 | 0.9834 | 0.7574 | shifted | Q1_good |
| ACK Flag Count | 0.6973 | 0.3611 | 0.6543 | 0.2777 | 0.9765 | 0.9507 | 0.3707 | shifted | Q4_underused_stable |
| Fwd Packet Length Mean | 0.6940 | 0.3575 | 0.6778 | 0.3438 | 0.9545 | 0.9050 | 0.4674 | shifted | Q1_good |
| Fwd Act Data Pkts | 0.6803 | 0.3353 | 0.7108 | 0.4121 | 0.7346 | 0.4427 | 0.1530 | shifted | Q4_underused_stable |
| Total TCP Flow Time | 0.6777 | 0.3179 | 0.6512 | 0.2734 | 0.9024 | 0.7972 | 0.5719 | shifted | Q4_underused_stable |
| CWR Flag Count | 0.6475 | 0.2698 | 0.6442 | 0.2700 | 0.7120 | 0.4071 | 0.0001 | restructured | Q3_noise |
| Fwd IAT Std | 0.6513 | 0.2663 | 0.6936 | 0.3663 | 0.9089 | 0.8131 | 0.0601 | restructured | Q3_noise |
| Bwd IAT Std | 0.6456 | 0.2584 | 0.6595 | 0.2921 | 0.8756 | 0.7413 | 0.1002 | restructured | Q3_noise |
| ECE Flag Count | 0.6405 | 0.2534 | 0.6366 | 0.2482 | 0.7117 | 0.4080 | 0.0000 | restructured | Q3_noise |
| PSH Flag Count | 0.6319 | 0.2234 | 0.6421 | 0.2615 | 0.7321 | 0.4302 | 0.1144 | shifted | Q1_good |
| Bwd PSH Flags | 0.6215 | 0.2179 | 0.6173 | 0.1929 | 0.7355 | 0.4485 | 0.1018 | restructured | Q2_fragile_shortcut |
| Down/Up Ratio | 0.6192 | 0.2071 | 0.6158 | 0.2077 | 0.7864 | 0.5552 | -0.0712 | flipped | Q2_fragile_shortcut |
| Fwd PSH Flags | 0.5939 | 0.1645 | 0.6186 | 0.2119 | 0.7319 | 0.4390 | 0.0991 | shifted | Q2_fragile_shortcut |
| RST Flag Count | 0.5943 | 0.1588 | 0.6443 | 0.2672 | 0.9094 | 0.8103 | -0.1053 | flipped | Q2_fragile_shortcut |
| Active Min | 0.5931 | 0.1547 | 0.6020 | 0.1752 | 0.5571 | 0.0963 | 0.0190 | shifted | Q3_noise |
| Bwd Packet Length Min | 0.5873 | 0.1298 | 0.6307 | 0.2410 | 0.5000 | -0.0017 | 0.2623 | shifted | Q1_good |
| SYN Flag Count | 0.5887 | 0.1213 | 0.6012 | 0.1719 | 0.6831 | 0.3448 | 0.2278 | restructured | Q1_good |
| Fwd RST Flags | 0.5696 | 0.1163 | 0.5958 | 0.1646 | 0.7313 | 0.4418 | -0.0689 | flipped | Q3_noise |
| Idle Mean | 0.5705 | 0.1107 | 0.5886 | 0.1501 | 0.5698 | 0.1222 | 0.0191 | shifted | Q3_noise |
| Fwd Packet Length Min | 0.5689 | 0.0963 | 0.6429 | 0.2632 | 0.5002 | -0.0029 | 0.2595 | restructured | Q1_good |
| Bwd Bulk Rate Avg | 0.5580 | 0.0944 | 0.5401 | 0.0740 | 0.6295 | 0.2347 | 0.0083 | collapsed | Q2_fragile_shortcut |
| Packet Length Min | 0.5688 | 0.0944 | 0.6242 | 0.2032 | 0.5000 | -0.0017 | 0.2594 | restructured | Q1_good |
| Bwd Bytes/Bulk Avg | 0.5507 | 0.0806 | 0.5137 | 0.0160 | 0.6310 | 0.2473 | 0.0089 | collapsed | Q3_noise |
| Active Max | 0.5385 | 0.0523 | 0.5472 | 0.0481 | 0.5571 | 0.0946 | 0.0143 | restructured | Q3_noise |
| Active Mean | 0.5350 | 0.0313 | 0.5594 | 0.0957 | 0.5506 | 0.0793 | 0.0127 | shifted | Q3_noise |
| Fwd Packet/Bulk Avg | 0.5212 | 0.0288 | 0.5222 | 0.0335 | 0.5043 | 0.0050 | 0.0002 | weak | Q3_noise |
| Fwd Bytes/Bulk Avg | 0.5187 | 0.0269 | 0.5213 | 0.0302 | 0.5100 | 0.0151 | 0.0003 | weak | Q2_fragile_shortcut |
| Bwd RST Flags | 0.5250 | 0.0254 | 0.5664 | 0.1066 | 0.6853 | 0.3579 | 0.0090 | collapsed | Q3_noise |
| Fwd Bulk Rate Avg | 0.5167 | 0.0250 | 0.5243 | 0.0326 | 0.5062 | 0.0073 | 0.0004 | weak | Q3_noise |
| ICMP Type | 0.5018 | 0.0035 | 0.5005 | -0.0023 | 0.5000 | 0.0000 | 0.0006 | weak | Q3_noise |
| Subflow Bwd Packets | 0.5010 | 0.0003 | 0.5012 | 0.0025 | 0.5000 | 0.0000 | 0.0000 | weak | Q3_noise |
| ICMP Code | 0.5000 | 0.0000 | 0.5005 | -0.0022 | 0.5000 | 0.0000 | 0.0006 | weak | Q2_fragile_shortcut |
| Fwd URG Flags | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | weak | Q3_noise |
| Bwd URG Flags | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | weak | Q3_noise |
| Subflow Fwd Packets | 0.5087 | -0.0017 | 0.5033 | -0.0063 | 0.5302 | 0.0426 | 0.0005 | weak | Q3_noise |
| Active Std | 0.5047 | -0.0061 | 0.5093 | -0.0220 | 0.5060 | 0.0053 | 0.0140 | stable | Q3_noise |
| FIN Flag Count | 0.5144 | -0.0189 | 0.5810 | 0.1201 | 0.8654 | 0.7226 | 0.0702 | stable | Q2_fragile_shortcut |
| Idle Std | 0.5046 | -0.0203 | 0.4980 | -0.0224 | 0.5040 | 0.0012 | 0.0137 | stable | Q3_noise |

_Per-feature C2ST-AUC vs its secondary shift metrics (Wasserstein-qn, MMD, KS, energy-distance, Anderson-Darling, all calibrated, with an agree/disagree flag) is in E1 (Supplementary checks, end of Step 11), not repeated here._

### Zero-fraction: all 71 features (sorted by |Δ zero-fraction| descending)

Per-feature zero/null rate from the step-7 profile, NOT a measure of drift by itself, just data-quality context (a feature with a high zero-rate makes C2ST/MMD/Wasserstein noisier); the DELTA column is the closest thing to a "did sparsity itself shift" signal. The zero-INFLATED-feature deep dive (zero fraction vs tail-only Wasserstein) is E2/E3, Supplementary checks, end of Step 11, this table is the plain per-feature listing for every feature, not just the zero-inflated subset.

| Feature | Zero-frac 2017 | Zero-frac 2018 | Δ Zero-frac | Verdict |
|---------|---------------:|---------------:|------------:|---------|
| ECE Flag Count | 0.9997 | 0.7086 | 0.2910 | restructured |
| CWR Flag Count | 0.9996 | 0.7086 | 0.2910 | restructured |
| Bwd IAT Min | 0.2517 | 0.4417 | 0.1900 | shifted |
| Fwd PSH Flags | 0.6239 | 0.4631 | 0.1608 | shifted |
| PSH Flag Count | 0.6219 | 0.4616 | 0.1604 | shifted |
| Bwd IAT Std | 0.6051 | 0.4576 | 0.1475 | restructured |
| Fwd Packet Length Std | 0.6034 | 0.4587 | 0.1448 | shifted |
| Bwd PSH Flags | 0.6293 | 0.4857 | 0.1437 | restructured |
| Fwd IAT Std | 0.5564 | 0.4210 | 0.1354 | restructured |
| Bwd IAT Max | 0.2463 | 0.3722 | 0.1259 | shifted |
| Bwd IAT Total | 0.2463 | 0.3722 | 0.1259 | shifted |
| Bwd IAT Mean | 0.2463 | 0.3722 | 0.1259 | restructured |
| Fwd IAT Max | 0.2218 | 0.3425 | 0.1207 | shifted |
| Fwd IAT Total | 0.2218 | 0.3425 | 0.1207 | shifted |
| Fwd IAT Mean | 0.2218 | 0.3425 | 0.1207 | shifted |
| Bwd Packet Length Std | 0.6086 | 0.4919 | 0.1167 | shifted |
| Fwd Act Data Pkts | 0.2775 | 0.3927 | 0.1152 | shifted |
| Flow IAT Std | 0.2346 | 0.3466 | 0.1120 | restructured |
| Fwd IAT Min | 0.2403 | 0.3477 | 0.1074 | flipped |
| Packet Length Variance | 0.1693 | 0.0705 | 0.0988 | shifted |
| Packet Length Std | 0.1693 | 0.0705 | 0.0988 | shifted |
| ACK Flag Count | 0.4946 | 0.4042 | 0.0904 | shifted |
| Fwd Packet Length Min | 0.5218 | 0.6045 | 0.0827 | restructured |
| Packet Length Min | 0.5232 | 0.6049 | 0.0816 | restructured |
| Packet Length Mean | 0.1439 | 0.0634 | 0.0806 | restructured |
| Flow Bytes/s | 0.1439 | 0.0634 | 0.0806 | restructured |
| Packet Length Max | 0.1439 | 0.0634 | 0.0806 | restructured |
| Fwd Packet Length Max | 0.1459 | 0.0654 | 0.0805 | flipped |
| Fwd Packet Length Mean | 0.1459 | 0.0654 | 0.0805 | shifted |
| Total Length of Fwd Packet | 0.1459 | 0.0654 | 0.0805 | shifted |
| FWD Init Win Bytes | 0.4814 | 0.4016 | 0.0797 | shifted |
| Total TCP Flow Time | 0.4768 | 0.3979 | 0.0789 | shifted |
| Bwd Bulk Rate Avg | 0.8611 | 0.9383 | 0.0772 | collapsed |
| Bwd Bytes/Bulk Avg | 0.8611 | 0.9383 | 0.0772 | collapsed |
| Bwd Packet Length Min | 0.5277 | 0.5988 | 0.0711 | shifted |
| Bwd Packet Length Mean | 0.1581 | 0.0940 | 0.0640 | shifted |
| Total Length of Bwd Packet | 0.1581 | 0.0940 | 0.0640 | restructured |
| Bwd Packet Length Max | 0.1581 | 0.0940 | 0.0640 | restructured |
| RST Flag Count | 0.7092 | 0.6486 | 0.0606 | flipped |
| Bwd Init Win Bytes | 0.6101 | 0.5496 | 0.0604 | restructured |
| SYN Flag Count | 0.5075 | 0.4550 | 0.0524 | restructured |
| FIN Flag Count | 0.6334 | 0.6757 | 0.0423 | stable |
| Flow IAT Min | 0.0387 | 0.0804 | 0.0417 | shifted |
| Fwd Packet/Bulk Avg | 0.9586 | 0.9935 | 0.0349 | weak |
| Fwd Bytes/Bulk Avg | 0.9586 | 0.9935 | 0.0349 | weak |
| Fwd Bulk Rate Avg | 0.9586 | 0.9935 | 0.0349 | weak |
| Fwd RST Flags | 0.8237 | 0.7915 | 0.0323 | flipped |
| Bwd RST Flags | 0.8772 | 0.8465 | 0.0306 | collapsed |
| Bwd Packets/s | 0.0262 | 0.0124 | 0.0138 | restructured |
| Total Bwd packets | 0.0262 | 0.0124 | 0.0138 | flipped |
| Subflow Fwd Packets | 0.9738 | 0.9876 | 0.0138 | weak |
| Down/Up Ratio | 0.0262 | 0.0133 | 0.0129 | flipped |
| Idle Std | 0.8951 | 0.8823 | 0.0128 | stable |
| Bwd Header Length | 0.0263 | 0.0137 | 0.0126 | restructured |
| Active Std | 0.8975 | 0.8851 | 0.0124 | stable |
| Active Min | 0.7808 | 0.7764 | 0.0044 | shifted |
| Active Mean | 0.7808 | 0.7764 | 0.0044 | shifted |
| Active Max | 0.7808 | 0.7764 | 0.0044 | restructured |
| Fwd Header Length | 0.0010 | 0.0038 | 0.0028 | shifted |
| Fwd Seg Size Min | 0.0045 | 0.0067 | 0.0022 | restructured |
| Idle Mean | 0.7776 | 0.7757 | 0.0018 | shifted |
| ICMP Code | 0.0001 | 0.0015 | 0.0014 | weak |
| Subflow Bwd Packets | 1.0000 | 0.9991 | 0.0009 | weak |
| Total Fwd Packet | 0.0000 | 0.0009 | 0.0009 | shifted |
| Fwd URG Flags | 1.0000 | 1.0000 | 0.0000 | weak |
| ICMP Type | 0.0000 | 0.0000 | 0.0000 | weak |
| Bwd URG Flags | 1.0000 | 1.0000 | 0.0000 | weak |
| Flow Packets/s | 0.0000 | 0.0000 | 0.0000 | restructured |
| Flow Duration | 0.0000 | 0.0000 | 0.0000 | restructured |
| Flow IAT Max | 0.0000 | 0.0000 | 0.0000 | shifted |
| Flow IAT Mean | 0.0000 | 0.0000 | 0.0000 | restructured |

### C2ST confidence intervals + routing: all 71 features (sorted by C2ST descending)

Per-feature 95% CI on C2ST-AUC (5-fold CV; wide at the individual-feature level, so the H1 claim rests on the aggregate Spearman, not on any one of these), shown both RAW and CALIBRATED (CI low/high run through the same pooled null-floor transform as the point estimate, see the Axis 1+2 summary table above for that null floor per feature), and the step-9 routing decision (`route`) that selected this feature's comparison template, joint shape between the two years: `nominal` (categorical PMF), `discrete_count` (count-shape metric), `continuous_unimodal` (single-mode Wasserstein/MMD), `continuous_multimodal` (same multimodal structure both years, drives the per-mode comparison below), or `structural_change` (the years' shapes do not share a template, e.g. cross-family or modality mismatch, so shape-agnostic C2ST is the arbiter).

| Feature | C2ST (raw) | CI low (raw) | CI high (raw) | C2ST (calibrated) | CI low (calibrated) | CI high (calibrated) | CI width (raw) | CV folds | Route |
|---------|---------:|-------:|--------:|---------:|-------:|--------:|---------:|---------:|-------|
| Bwd IAT Min | 0.8502 | 0.8431 | 0.8574 | 0.6857 | 0.6707 | 0.7007 | 0.0143 | [0.8434625, 0.8458812499999999, 0.857475, 0.85411875, 0.8501468750000001] | continuous_multimodal |
| Flow IAT Max | 0.8279 | 0.8180 | 0.8377 | 0.6376 | 0.6169 | 0.6584 | 0.0197 | [0.8241531249999999, 0.8408281249999999, 0.8224749999999998, 0.83005625, 0.82186875] | continuous_multimodal |
| Flow Duration | 0.8228 | 0.8097 | 0.8358 | 0.6345 | 0.6075 | 0.6614 | 0.0261 | [0.8155593750000001, 0.827184375, 0.8112468749999999, 0.838184375, 0.8216937500000001] | structural_change |
| Bwd Packets/s | 0.8222 | 0.8083 | 0.8361 | 0.6260 | 0.5968 | 0.6553 | 0.0278 | [0.8052375, 0.8321968750000001, 0.831953125, 0.8181437500000001, 0.823653125] | structural_change |
| Flow Packets/s | 0.8147 | 0.7895 | 0.8399 | 0.6147 | 0.5623 | 0.6670 | 0.0504 | [0.8379593750000001, 0.7997375, 0.8315874999999999, 0.813771875, 0.79030625] | structural_change |
| Flow IAT Min | 0.8066 | 0.7840 | 0.8292 | 0.5995 | 0.5526 | 0.6463 | 0.0452 | [0.798509375, 0.82368125, 0.800046875, 0.8269124999999999, 0.7839031250000001] | continuous_multimodal |
| Fwd IAT Mean | 0.7991 | 0.7807 | 0.8175 | 0.5833 | 0.5451 | 0.6215 | 0.0368 | [0.8187125, 0.782990625, 0.78600625, 0.8007187499999999, 0.80695625] | continuous_multimodal |
| Fwd IAT Min | 0.7981 | 0.7759 | 0.8204 | 0.5811 | 0.5349 | 0.6273 | 0.0445 | [0.80775, 0.77806875, 0.802909375, 0.7816124999999999, 0.8203875] | continuous_multimodal |
| Fwd IAT Max | 0.7923 | 0.7825 | 0.8021 | 0.5768 | 0.5569 | 0.5967 | 0.0196 | [0.800434375, 0.799328125, 0.7833937500000001, 0.7931562499999999, 0.7850593749999999] | continuous_multimodal |
| Flow IAT Mean | 0.7904 | 0.7734 | 0.8074 | 0.5703 | 0.5355 | 0.6052 | 0.0340 | [0.78059375, 0.7747187500000001, 0.8099687499999999, 0.7916187499999999, 0.7951750000000001] | structural_change |
| Total Length of Fwd Packet | 0.7906 | 0.7770 | 0.8042 | 0.5533 | 0.5243 | 0.5822 | 0.0271 | [0.8009218749999999, 0.7961843749999999, 0.778965625, 0.7983187500000001, 0.7785875] | continuous_multimodal |
| Packet Length Variance | 0.7850 | 0.7569 | 0.8131 | 0.5492 | 0.4904 | 0.6081 | 0.0562 | [0.782309375, 0.7886625, 0.7522625, 0.7859281250000001, 0.8158593749999999] | continuous_multimodal |
| Packet Length Max | 0.7848 | 0.7676 | 0.8020 | 0.5483 | 0.5123 | 0.5844 | 0.0344 | [0.7711375, 0.783334375, 0.7715562499999999, 0.79953125, 0.798425] | structural_change |
| Bwd Header Length | 0.7834 | 0.7603 | 0.8065 | 0.5460 | 0.4977 | 0.5944 | 0.0462 | [0.76976875, 0.76928125, 0.814853125, 0.7824562499999999, 0.7807] | structural_change |
| Bwd Packet Length Max | 0.7832 | 0.7586 | 0.8077 | 0.5435 | 0.4919 | 0.5951 | 0.0490 | [0.813690625, 0.7839656250000001, 0.7655812500000001, 0.786803125, 0.7657187500000001] | structural_change |
| Bwd IAT Max | 0.7768 | 0.7595 | 0.7942 | 0.5428 | 0.5073 | 0.5784 | 0.0347 | [0.79573125, 0.764159375, 0.76775, 0.7876375, 0.7687812500000001] | continuous_multimodal |
| Total Length of Bwd Packet | 0.7772 | 0.7543 | 0.8002 | 0.5398 | 0.4925 | 0.5872 | 0.0459 | [0.75160625, 0.7887875000000001, 0.76376875, 0.7881343749999999, 0.793790625] | structural_change |
| Fwd IAT Total | 0.7780 | 0.7648 | 0.7913 | 0.5383 | 0.5107 | 0.5659 | 0.0265 | [0.787821875, 0.7892625, 0.7779437499999999, 0.7649656250000001, 0.770096875] | continuous_multimodal |
| Fwd Header Length | 0.7800 | 0.7571 | 0.8029 | 0.5315 | 0.4827 | 0.5803 | 0.0458 | [0.747075, 0.7895343750000001, 0.78748125, 0.78953125, 0.7863093750000001] | continuous_multimodal |
| Packet Length Std | 0.7679 | 0.7525 | 0.7833 | 0.5188 | 0.4869 | 0.5507 | 0.0308 | [0.7486437499999999, 0.7784749999999999, 0.778515625, 0.769846875, 0.7641000000000001] | continuous_multimodal |
| Bwd IAT Mean | 0.7560 | 0.7385 | 0.7734 | 0.4907 | 0.4543 | 0.5272 | 0.0349 | [0.7503343749999999, 0.7564, 0.7684375, 0.7354031250000002, 0.7693875] | structural_change |
| FWD Init Win Bytes | 0.7558 | 0.7457 | 0.7658 | 0.4863 | 0.4652 | 0.5075 | 0.0201 | [0.753625, 0.7635750000000001, 0.7428874999999999, 0.7609000000000001, 0.7579343749999999] | continuous_multimodal |
| Bwd Init Win Bytes | 0.7365 | 0.7190 | 0.7540 | 0.4602 | 0.4244 | 0.4961 | 0.0350 | [0.7378875, 0.7258875, 0.7300625000000001, 0.728096875, 0.76033125] | structural_change |
| Bwd Packet Length Std | 0.7308 | 0.7096 | 0.7521 | 0.4368 | 0.3923 | 0.4813 | 0.0426 | [0.7344875, 0.73541875, 0.708140625, 0.7542, 0.72191875] | continuous_multimodal |
| Flow Bytes/s | 0.7319 | 0.7150 | 0.7488 | 0.4308 | 0.3949 | 0.4666 | 0.0338 | [0.728390625, 0.729215625, 0.7545000000000001, 0.7175875, 0.729590625] | structural_change |
| Total Bwd packets | 0.7243 | 0.7075 | 0.7412 | 0.4235 | 0.3882 | 0.4588 | 0.0338 | [0.7273406250000001, 0.7134156249999999, 0.71111875, 0.74524375, 0.7245906249999998] | continuous_multimodal |
| Bwd IAT Total | 0.7224 | 0.7170 | 0.7277 | 0.4190 | 0.4078 | 0.4302 | 0.0107 | [0.7276875, 0.7165468749999999, 0.72216875, 0.725146875, 0.7203499999999999] | continuous_multimodal |
| Fwd Packet Length Max | 0.7243 | 0.7033 | 0.7452 | 0.4167 | 0.3723 | 0.4610 | 0.0420 | [0.731090625, 0.712834375, 0.7501874999999999, 0.7193125, 0.7078874999999999] | structural_change |
| Packet Length Mean | 0.7169 | 0.7005 | 0.7333 | 0.4011 | 0.3664 | 0.4358 | 0.0328 | [0.730003125, 0.7083999999999999, 0.7298187500000001, 0.6999562500000001, 0.71626875] | structural_change |
| Flow IAT Std | 0.7073 | 0.6872 | 0.7274 | 0.3960 | 0.3546 | 0.4375 | 0.0402 | [0.72045, 0.7051499999999999, 0.682578125, 0.7232218750000001, 0.7051000000000001] | structural_change |
| Total Fwd Packet | 0.7059 | 0.6955 | 0.7164 | 0.3950 | 0.3735 | 0.4164 | 0.0208 | [0.704459375, 0.7146125, 0.697946875, 0.697971875, 0.7147125] | continuous_multimodal |
| Fwd Packet Length Std | 0.6927 | 0.6670 | 0.7183 | 0.3715 | 0.3190 | 0.4240 | 0.0513 | [0.6845125000000001, 0.7114093749999999, 0.7180468749999999, 0.674159375, 0.6752406249999999] | continuous_multimodal |
| Fwd Seg Size Min | 0.7001 | 0.6858 | 0.7143 | 0.3705 | 0.3406 | 0.4003 | 0.0285 | [0.7069875, 0.715746875, 0.6976874999999999, 0.6865625, 0.6934187500000001] | structural_change |
| Bwd Packet Length Mean | 0.6969 | 0.6706 | 0.7231 | 0.3681 | 0.3134 | 0.4228 | 0.0525 | [0.686253125, 0.695253125, 0.730634375, 0.673859375, 0.69829375] | continuous_multimodal |
| ACK Flag Count | 0.6973 | 0.6735 | 0.7211 | 0.3611 | 0.3109 | 0.4113 | 0.0476 | [0.697128125, 0.71023125, 0.68395625, 0.6740093749999998, 0.7214218749999998] | continuous_multimodal |
| Fwd Packet Length Mean | 0.6940 | 0.6800 | 0.7081 | 0.3575 | 0.3280 | 0.3870 | 0.0281 | [0.676015625, 0.7049375, 0.7006937500000001, 0.690840625, 0.697721875] | continuous_multimodal |
| Fwd Act Data Pkts | 0.6803 | 0.6677 | 0.6928 | 0.3353 | 0.3092 | 0.3613 | 0.0251 | [0.6657843750000001, 0.685825, 0.6746343750000002, 0.691390625, 0.683690625] | continuous_multimodal |
| Total TCP Flow Time | 0.6777 | 0.6634 | 0.6921 | 0.3179 | 0.2875 | 0.3483 | 0.0287 | [0.6778562499999999, 0.665259375, 0.693165625, 0.6845031250000001, 0.6679218750000001] | continuous_multimodal |
| CWR Flag Count | 0.6475 | 0.6319 | 0.6631 | 0.2698 | 0.2376 | 0.3020 | 0.0311 | [0.6375, 0.65125, 0.6675, 0.64375, 0.6375] | structural_change |
| Fwd IAT Std | 0.6513 | 0.6323 | 0.6704 | 0.2663 | 0.2261 | 0.3065 | 0.0382 | [0.649946875, 0.6691843749999999, 0.6387187500000001, 0.66459375, 0.6342968749999999] | structural_change |
| Bwd IAT Std | 0.6456 | 0.6413 | 0.6499 | 0.2584 | 0.2494 | 0.2674 | 0.0086 | [0.6397499999999999, 0.647734375, 0.648, 0.64500625, 0.64735625] | structural_change |
| ECE Flag Count | 0.6405 | 0.6254 | 0.6556 | 0.2534 | 0.2221 | 0.2847 | 0.0301 | [0.63375, 0.64375, 0.625, 0.6425, 0.6575] | structural_change |
| PSH Flag Count | 0.6319 | 0.6117 | 0.6522 | 0.2234 | 0.1807 | 0.2661 | 0.0405 | [0.6208031249999999, 0.6249062500000001, 0.620228125, 0.659259375, 0.6343656249999999] | continuous_multimodal |
| Bwd PSH Flags | 0.6215 | 0.6053 | 0.6376 | 0.2179 | 0.1845 | 0.2513 | 0.0324 | [0.612896875, 0.6190375, 0.6226718750000001, 0.6429312500000001, 0.6097499999999999] | structural_change |
| Down/Up Ratio | 0.6192 | 0.5975 | 0.6409 | 0.2071 | 0.1619 | 0.2522 | 0.0434 | [0.6421875000000001, 0.6059875, 0.6283249999999998, 0.6210500000000001, 0.59841875] | structural_change |
| Fwd PSH Flags | 0.5939 | 0.5714 | 0.6163 | 0.1645 | 0.1183 | 0.2107 | 0.0449 | [0.6234843749999999, 0.5746812499999999, 0.594053125, 0.58685625, 0.590221875] | continuous_multimodal |
| RST Flag Count | 0.5943 | 0.5811 | 0.6074 | 0.1588 | 0.1316 | 0.1860 | 0.0262 | [0.5849031250000001, 0.5997093750000001, 0.60969375, 0.58499375, 0.59196875] | discrete_count |
| Active Min | 0.5931 | 0.5799 | 0.6064 | 0.1547 | 0.1272 | 0.1823 | 0.0265 | [0.583315625, 0.6037625, 0.6055812499999998, 0.5851968750000001, 0.5877468749999999] | continuous_multimodal |
| Bwd Packet Length Min | 0.5873 | 0.5754 | 0.5993 | 0.1298 | 0.1047 | 0.1549 | 0.0238 | [0.593925, 0.583959375, 0.597909375, 0.573125, 0.5877531250000001] | continuous_multimodal |
| SYN Flag Count | 0.5887 | 0.5611 | 0.6163 | 0.1213 | 0.0624 | 0.1802 | 0.0551 | [0.5896312499999998, 0.593953125, 0.594434375, 0.61305, 0.552396875] | structural_change |
| Fwd RST Flags | 0.5696 | 0.5597 | 0.5796 | 0.1163 | 0.0959 | 0.1368 | 0.0200 | [0.5807625, 0.5597, 0.570721875, 0.5645, 0.5725218750000001] | discrete_count |
| Idle Mean | 0.5705 | 0.5551 | 0.5858 | 0.1107 | 0.0789 | 0.1425 | 0.0307 | [0.5552062500000001, 0.574415625, 0.5651749999999999, 0.5887249999999998, 0.5689031250000001] | continuous_multimodal |
| Fwd Packet Length Min | 0.5689 | 0.5486 | 0.5892 | 0.0963 | 0.0537 | 0.1389 | 0.0407 | [0.5813312500000001, 0.5738750000000001, 0.572928125, 0.5761281249999999, 0.540175] | structural_change |
| Bwd Bulk Rate Avg | 0.5580 | 0.5441 | 0.5718 | 0.0944 | 0.0661 | 0.1228 | 0.0277 | [0.5497656249999999, 0.56211875, 0.553634375, 0.5487625, 0.57555] | continuous_multimodal |
| Packet Length Min | 0.5688 | 0.5539 | 0.5836 | 0.0944 | 0.0631 | 0.1256 | 0.0298 | [0.5754281250000001, 0.5572843749999999, 0.5548687499999999, 0.5741406249999998, 0.58203125] | structural_change |
| Bwd Bytes/Bulk Avg | 0.5507 | 0.5361 | 0.5654 | 0.0806 | 0.0505 | 0.1106 | 0.0293 | [0.5456875, 0.5438625, 0.5386031250000001, 0.5677249999999999, 0.5578187499999999] | structural_change |
| Active Max | 0.5385 | 0.5275 | 0.5495 | 0.0523 | 0.0297 | 0.0748 | 0.0219 | [0.5529156250000001, 0.5322375000000001, 0.536925, 0.5396937500000001, 0.53066875] | structural_change |
| Active Mean | 0.5350 | 0.5233 | 0.5466 | 0.0313 | 0.0070 | 0.0556 | 0.0233 | [0.537075, 0.54653125, 0.5395375, 0.5221093750000001, 0.529653125] | continuous_multimodal |
| Fwd Packet/Bulk Avg | 0.5212 | 0.5170 | 0.5255 | 0.0288 | 0.0202 | 0.0374 | 0.0085 | [0.52626875, 0.5224562500000001, 0.5187437500000001, 0.5175000000000001, 0.521228125] | structural_change |
| Fwd Bytes/Bulk Avg | 0.5187 | 0.5073 | 0.5302 | 0.0269 | 0.0038 | 0.0500 | 0.0229 | [0.533734375, 0.51001875, 0.5186562499999999, 0.512571875, 0.51869375] | continuous_multimodal |
| Bwd RST Flags | 0.5250 | 0.5098 | 0.5403 | 0.0254 | 0.0000 | 0.0567 | 0.0305 | [0.5217593749999999, 0.5466749999999999, 0.51596875, 0.520640625, 0.5201875] | discrete_count |
| Fwd Bulk Rate Avg | 0.5167 | 0.5073 | 0.5262 | 0.0250 | 0.0059 | 0.0441 | 0.0189 | [0.516209375, 0.50995, 0.50875, 0.522440625, 0.5261875] | continuous_multimodal |
| ICMP Type | 0.5018 | 0.5004 | 0.5031 | 0.0035 | 0.0007 | 0.0063 | 0.0028 | [0.50375, 0.50125, 0.50125, 0.50125, 0.50125] | nominal |
| Subflow Bwd Packets | 0.5010 | 0.4993 | 0.5027 | 0.0003 | 0.0000 | 0.0037 | 0.0034 | [0.5, 0.5025, 0.5, 0.5025, 0.5] | nominal |
| ICMP Code | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0.5, 0.5, 0.5, 0.5, 0.5] | nominal |
| Fwd URG Flags | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0.5, 0.5, 0.5, 0.5, 0.5] | nominal |
| Bwd URG Flags | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0.5, 0.5, 0.5, 0.5, 0.5] | nominal |
| Subflow Fwd Packets | 0.5087 | 0.5048 | 0.5127 | -0.0017 | 0.0000 | 0.0063 | 0.0079 | [0.51125, 0.5125, 0.5087499999999999, 0.505, 0.50625] | nominal |
| Active Std | 0.5047 | 0.4927 | 0.5166 | -0.0061 | 0.0000 | 0.0182 | 0.0239 | [0.48974375000000003, 0.512375, 0.514, 0.50375, 0.5035343750000001] | structural_change |
| FIN Flag Count | 0.5144 | 0.5002 | 0.5285 | -0.0189 | 0.0000 | 0.0108 | 0.0284 | [0.51545, 0.50468125, 0.50479375, 0.53266875, 0.5141625000000001] | discrete_count |
| Idle Std | 0.5046 | 0.4913 | 0.5178 | -0.0203 | 0.0000 | 0.0070 | 0.0265 | [0.490403125, 0.50044375, 0.504946875, 0.5197437500000001, 0.5072625] | structural_change |

A handful of other step-10 robustness/routing fields are computed per feature but not tabulated above since they are secondary robustness checks rather than headline results: `separation_stability_auc` (legacy folded-AUC-only version of Axis 2, no MI term), `null_separation_threshold_2017/2018` and `separation_strong_effective_2017/2018` (the per-feature null floor a feature must clear, at least max(0.55, this value), to count as separated, feeding the flip/collapse verdict and Axis-2 status above). Step 10 also stores the E1 cross-metric agreement per feature (`e1_agreement`, for each corroboration metric, does its own null-calibrated shifted/stable vote match the C2ST verdict?, and `e1_agreement_rate`, the fraction that do). E1 is summarized in the Supplementary checks at the end of Step 11.

### Per-attack-family breakdown: Axis 1 shift and Axis 2 stability, all features

Step 10 computes Axis 1 (per-family slice C2ST-AUC, slices carry C2ST only; the corroboration distances are pooled-only by design) and Axis 2 (separation stability) separately for EACH attack family, not just pooled benign-vs-attack. Axis 2 needs no calibration (it is not a C2ST-family metric); Axis 1's per-family C2ST is shown both raw and calibrated against that family's OWN null floor (smaller families are noisier, hence a higher floor). Duplicated here in full, the same underlying numbers are also rendered, pivoted differently, in Section 1's C2b feature x family matrix further below under Step 11.

**Axis 1: per-family C2ST-AUC, RAW** (71 features x 6 families)

| Feature | Botnet | BruteForce | DDoS | DoS | Infiltration | WebAttack |
|---|---:|---:|---:|---:|---:|---:|
| ACK Flag Count | 0.9979 | 0.9995 | 0.9990 | 0.9580 | 0.8958 | 0.8958 |
| Active Max | 0.5000 | 0.5005 | 0.7386 | 0.5342 | 0.5280 | 0.5190 |
| Active Mean | 0.5000 | 0.5000 | 0.7323 | 0.5302 | 0.5316 | 0.5190 |
| Active Min | 0.5000 | 0.5000 | 0.7314 | 0.5400 | 0.5368 | 0.5190 |
| Active Std | 0.5000 | 0.5000 | 0.5000 | 0.5191 | 0.5045 | 0.5000 |
| Bwd Bulk Rate Avg | 0.5000 | 0.5018 | 0.6028 | 0.8295 | 0.5000 | 0.5123 |
| Bwd Bytes/Bulk Avg | 0.5000 | 0.5028 | 0.6025 | 0.8253 | 0.5000 | 0.5123 |
| Bwd Header Length | 0.9460 | 0.9991 | 0.7907 | 0.9243 | 0.8980 | 0.9821 |
| Bwd IAT Max | 0.9925 | 1.0000 | 0.8157 | 0.9172 | 0.7004 | 0.9206 |
| Bwd IAT Mean | 0.9776 | 1.0000 | 0.8509 | 0.7597 | 0.7017 | 0.9671 |
| Bwd IAT Min | 0.9894 | 0.9895 | 0.9876 | 0.9448 | 0.6929 | 0.9291 |
| Bwd IAT Std | 0.9909 | 1.0000 | 0.8201 | 0.8835 | 0.5410 | 0.9337 |
| Bwd IAT Total | 0.9920 | 1.0000 | 0.8211 | 0.8313 | 0.6922 | 0.9550 |
| Bwd Init Win Bytes | 0.9965 | 0.9998 | 1.0000 | 0.9996 | 0.5186 | 0.9772 |
| Bwd PSH Flags | 0.9524 | 0.7913 | 0.5250 | 0.5163 | 0.5267 | 0.4849 |
| Bwd Packet Length Max | 1.0000 | 0.7888 | 1.0000 | 0.9994 | 0.5244 | 0.8112 |
| Bwd Packet Length Mean | 0.9961 | 0.9995 | 1.0000 | 0.9992 | 0.5244 | 0.9269 |
| Bwd Packet Length Min | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5025 | 0.5000 |
| Bwd Packet Length Std | 0.9953 | 0.9995 | 1.0000 | 0.9994 | 0.5247 | 0.9506 |
| Bwd Packets/s | 0.9740 | 1.0000 | 0.9533 | 0.7586 | 0.9798 | 0.9646 |
| Bwd RST Flags | 0.5000 | 0.7790 | 0.5000 | 0.5050 | 0.8758 | 0.5000 |
| Bwd URG Flags | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| CWR Flag Count | 1.0000 | 0.5000 | 0.9985 | 0.5000 | 0.5355 | 1.0000 |
| Down/Up Ratio | 0.9986 | 0.9977 | 0.9934 | 0.8680 | 0.7195 | 0.9007 |
| ECE Flag Count | 1.0000 | 0.5000 | 0.9990 | 0.5000 | 0.5330 | 1.0000 |
| FIN Flag Count | 0.5000 | 0.5030 | 0.7540 | 0.8372 | 0.5278 | 0.5000 |
| FWD Init Win Bytes | 0.5000 | 1.0000 | 0.8969 | 1.0000 | 0.5486 | 1.0000 |
| Flow Bytes/s | 0.9918 | 1.0000 | 0.9695 | 0.9601 | 0.5305 | 0.9583 |
| Flow Duration | 0.9932 | 1.0000 | 0.9504 | 0.8423 | 0.9772 | 0.9479 |
| Flow IAT Max | 0.9933 | 1.0000 | 0.9495 | 0.9111 | 0.9797 | 0.9221 |
| Flow IAT Mean | 0.9742 | 1.0000 | 0.9662 | 0.7986 | 0.9815 | 0.8927 |
| Flow IAT Min | 0.9366 | 0.8383 | 0.7884 | 0.9782 | 0.9307 | 0.7359 |
| Flow IAT Std | 0.9918 | 1.0000 | 0.9590 | 0.8905 | 0.7062 | 0.9092 |
| Flow Packets/s | 0.9734 | 1.0000 | 0.9591 | 0.8013 | 0.9832 | 0.8877 |
| Fwd Act Data Pkts | 0.5288 | 0.7913 | 0.5018 | 0.5119 | 0.5243 | 0.4600 |
| Fwd Bulk Rate Avg | 0.5271 | 0.5005 | 0.5010 | 0.5149 | 0.5017 | 0.5000 |
| Fwd Bytes/Bulk Avg | 0.5313 | 0.5015 | 0.5000 | 0.5144 | 0.5010 | 0.5000 |
| Fwd Header Length | 0.5748 | 0.9479 | 1.0000 | 0.9608 | 0.8909 | 0.9804 |
| Fwd IAT Max | 0.9953 | 1.0000 | 0.9544 | 0.9037 | 0.8872 | 0.9124 |
| Fwd IAT Mean | 0.9657 | 1.0000 | 0.9674 | 0.8001 | 0.8886 | 0.9566 |
| Fwd IAT Min | 0.8498 | 0.9108 | 0.9434 | 0.9857 | 0.8992 | 0.9479 |
| Fwd IAT Std | 0.9921 | 1.0000 | 0.9621 | 0.8684 | 0.5387 | 0.9044 |
| Fwd IAT Total | 0.9967 | 1.0000 | 0.9595 | 0.8497 | 0.8881 | 0.9455 |
| Fwd PSH Flags | 0.5000 | 0.7960 | 0.5015 | 0.5003 | 0.5284 | 0.4600 |
| Fwd Packet Length Max | 0.9998 | 0.7857 | 0.8917 | 0.5246 | 0.5277 | 0.9833 |
| Fwd Packet Length Mean | 0.9998 | 0.9939 | 0.9998 | 0.9166 | 0.5294 | 0.9683 |
| Fwd Packet Length Min | 0.5000 | 0.5000 | 0.5005 | 0.5028 | 0.5035 | 0.5000 |
| Fwd Packet Length Std | 1.0000 | 0.9962 | 0.9998 | 0.8464 | 0.5295 | 0.9488 |
| Fwd Packet/Bulk Avg | 0.5254 | 0.5010 | 0.5010 | 0.5158 | 0.5035 | 0.5000 |
| Fwd RST Flags | 0.5000 | 0.5000 | 0.8832 | 0.9668 | 0.5008 | 0.5000 |
| Fwd Seg Size Min | 0.5000 | 0.5000 | 0.5040 | 0.7458 | 0.5535 | 1.0000 |
| Fwd URG Flags | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5030 | 0.5000 |
| ICMP Code | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4990 | 0.5000 |
| ICMP Type | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5015 | 0.5000 |
| Idle Mean | 0.5000 | 0.5000 | 0.7840 | 0.5392 | 0.5267 | 0.5253 |
| Idle Std | 0.5000 | 0.5000 | 0.5000 | 0.5197 | 0.5149 | 0.5000 |
| PSH Flag Count | 0.9524 | 0.7993 | 0.5237 | 0.5153 | 0.5279 | 0.4849 |
| Packet Length Max | 1.0000 | 0.7845 | 1.0000 | 0.9997 | 0.5247 | 0.7915 |
| Packet Length Mean | 1.0000 | 0.9975 | 1.0000 | 0.9973 | 0.5323 | 0.9629 |
| Packet Length Min | 0.5000 | 0.5000 | 0.5005 | 0.5000 | 0.5050 | 0.5000 |
| Packet Length Std | 1.0000 | 0.9990 | 1.0000 | 0.9975 | 0.5269 | 0.9560 |
| Packet Length Variance | 0.9998 | 0.9995 | 0.9998 | 0.9969 | 0.5258 | 0.9560 |
| RST Flag Count | 0.5000 | 0.7835 | 0.8990 | 0.9769 | 0.8663 | 0.5000 |
| SYN Flag Count | 0.5000 | 0.5002 | 0.5020 | 0.5438 | 0.8502 | 0.5000 |
| Subflow Bwd Packets | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Subflow Fwd Packets | 0.5000 | 0.5000 | 0.5012 | 0.5000 | 0.7015 | 0.5000 |
| Total Bwd packets | 0.9460 | 1.0000 | 0.7868 | 0.9276 | 0.8889 | 0.5250 |
| Total Fwd Packet | 0.5754 | 0.9441 | 0.9995 | 0.9694 | 0.8778 | 0.9036 |
| Total Length of Bwd Packet | 0.9998 | 0.9998 | 1.0000 | 0.9997 | 0.5266 | 0.9727 |
| Total Length of Fwd Packet | 0.9972 | 0.9945 | 0.8892 | 0.5300 | 0.5325 | 0.9712 |
| Total TCP Flow Time | 0.9918 | 1.0000 | 0.9523 | 0.8335 | 0.9772 | 0.9479 |

**Axis 1: per-family C2ST-AUC, CALIBRATED** (71 features x 6 families)

| Feature | Botnet | BruteForce | DDoS | DoS | Infiltration | WebAttack |
|---|---:|---:|---:|---:|---:|---:|
| ACK Flag Count | 0.9956 | 0.9989 | 0.9978 | 0.9119 | 0.7828 | 0.7159 |
| Active Max | 0.0000 | 0.0010 | 0.4530 | 0.0526 | 0.0451 | -0.0404 |
| Active Mean | -0.0017 | 0.0000 | 0.4461 | 0.0450 | 0.0484 | -0.0642 |
| Active Min | 0.0000 | 0.0000 | 0.4430 | 0.0709 | 0.0605 | -0.0766 |
| Active Std | 0.0000 | 0.0000 | 0.0000 | 0.0233 | 0.0021 | 0.0000 |
| Bwd Bulk Rate Avg | 0.0000 | -0.0000 | 0.1714 | 0.6466 | 0.0000 | -0.0182 |
| Bwd Bytes/Bulk Avg | 0.0000 | 0.0005 | 0.1891 | 0.6395 | 0.0000 | -0.0102 |
| Bwd Header Length | 0.8887 | 0.9981 | 0.5527 | 0.8423 | 0.7849 | 0.9596 |
| Bwd IAT Max | 0.9842 | 1.0000 | 0.6175 | 0.8295 | 0.3879 | 0.8201 |
| Bwd IAT Mean | 0.9529 | 1.0000 | 0.6856 | 0.4981 | 0.3875 | 0.9237 |
| Bwd IAT Min | 0.9773 | 0.9778 | 0.9741 | 0.8866 | 0.3677 | 0.8426 |
| Bwd IAT Std | 0.9805 | 1.0000 | 0.6323 | 0.7574 | 0.0722 | 0.8498 |
| Bwd IAT Total | 0.9828 | 1.0000 | 0.6279 | 0.6508 | 0.3645 | 0.8928 |
| Bwd Init Win Bytes | 0.9929 | 0.9995 | 1.0000 | 0.9993 | 0.0300 | 0.9395 |
| Bwd PSH Flags | 0.8983 | 0.5675 | 0.0344 | 0.0140 | 0.0358 | -0.2208 |
| Bwd Packet Length Max | 1.0000 | 0.5519 | 1.0000 | 0.9988 | 0.0366 | 0.5702 |
| Bwd Packet Length Mean | 0.9918 | 0.9990 | 1.0000 | 0.9983 | 0.0409 | 0.8371 |
| Bwd Packet Length Min | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 | 0.0000 |
| Bwd Packet Length Std | 0.9903 | 0.9990 | 1.0000 | 0.9988 | 0.0347 | 0.8878 |
| Bwd Packets/s | 0.9458 | 1.0000 | 0.9044 | 0.5078 | 0.9581 | 0.9045 |
| Bwd RST Flags | 0.0000 | 0.5349 | 0.0000 | 0.0053 | 0.7428 | 0.0000 |
| Bwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CWR Flag Count | 1.0000 | 0.0000 | 0.9969 | 0.0000 | 0.0597 | 1.0000 |
| Down/Up Ratio | 0.9970 | 0.9951 | 0.9862 | 0.7256 | 0.4202 | 0.7412 |
| ECE Flag Count | 1.0000 | 0.0000 | 0.9979 | 0.0000 | 0.0548 | 1.0000 |
| FIN Flag Count | 0.0000 | -0.0005 | 0.4856 | 0.6622 | 0.0395 | -0.0132 |
| FWD Init Win Bytes | 0.0000 | 1.0000 | 0.7870 | 1.0000 | 0.0736 | 1.0000 |
| Flow Bytes/s | 0.9826 | 1.0000 | 0.9376 | 0.9159 | 0.0468 | 0.8970 |
| Flow Duration | 0.9859 | 1.0000 | 0.8950 | 0.6678 | 0.9523 | 0.8735 |
| Flow IAT Max | 0.9859 | 1.0000 | 0.8933 | 0.8160 | 0.9571 | 0.8114 |
| Flow IAT Mean | 0.9455 | 1.0000 | 0.9296 | 0.5820 | 0.9618 | 0.7562 |
| Flow IAT Min | 0.8643 | 0.6627 | 0.5680 | 0.9535 | 0.8508 | 0.3655 |
| Flow IAT Std | 0.9828 | 1.0000 | 0.9154 | 0.7739 | 0.3903 | 0.7725 |
| Flow Packets/s | 0.9439 | 1.0000 | 0.9142 | 0.5922 | 0.9655 | 0.7012 |
| Fwd Act Data Pkts | 0.0384 | 0.5654 | 0.0018 | 0.0023 | 0.0312 | -0.1513 |
| Fwd Bulk Rate Avg | 0.0348 | -0.0024 | 0.0003 | 0.0135 | -0.0018 | 0.0000 |
| Fwd Bytes/Bulk Avg | 0.0428 | -0.0005 | 0.0000 | 0.0173 | -0.0029 | 0.0000 |
| Fwd Header Length | 0.1311 | 0.8913 | 1.0000 | 0.9184 | 0.7788 | 0.9557 |
| Fwd IAT Max | 0.9902 | 1.0000 | 0.9055 | 0.8017 | 0.7671 | 0.7816 |
| Fwd IAT Mean | 0.9266 | 1.0000 | 0.9309 | 0.5756 | 0.7736 | 0.9025 |
| Fwd IAT Min | 0.6776 | 0.8169 | 0.8778 | 0.9703 | 0.7915 | 0.8742 |
| Fwd IAT Std | 0.9834 | 1.0000 | 0.9217 | 0.7251 | 0.0719 | 0.7810 |
| Fwd IAT Total | 0.9930 | 1.0000 | 0.9165 | 0.6905 | 0.7705 | 0.8645 |
| Fwd PSH Flags | 0.0000 | 0.5764 | -0.0003 | -0.0249 | 0.0439 | -0.2137 |
| Fwd Packet Length Max | 0.9995 | 0.5557 | 0.7750 | -0.0062 | 0.0280 | 0.9600 |
| Fwd Packet Length Mean | 0.9995 | 0.9873 | 0.9995 | 0.8290 | 0.0525 | 0.9309 |
| Fwd Packet Length Min | 0.0000 | 0.0000 | -0.0007 | 0.0018 | 0.0020 | 0.0000 |
| Fwd Packet Length Std | 1.0000 | 0.9921 | 0.9995 | 0.6792 | 0.0469 | 0.8832 |
| Fwd Packet/Bulk Avg | 0.0386 | 0.0003 | 0.0004 | 0.0182 | 0.0037 | 0.0000 |
| Fwd RST Flags | 0.0000 | 0.0000 | 0.7576 | 0.9293 | -0.0002 | -0.0138 |
| Fwd Seg Size Min | 0.0000 | -0.0017 | 0.0029 | 0.4643 | 0.0883 | 1.0000 |
| Fwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0006 | 0.0000 |
| ICMP Code | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0038 | 0.0000 |
| ICMP Type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0003 | 0.0000 |
| Idle Mean | 0.0000 | 0.0000 | 0.5563 | 0.0616 | 0.0375 | -0.0394 |
| Idle Std | 0.0000 | 0.0000 | 0.0000 | 0.0246 | 0.0217 | 0.0000 |
| PSH Flag Count | 0.9007 | 0.5774 | 0.0301 | 0.0089 | 0.0368 | -0.1767 |
| Packet Length Max | 1.0000 | 0.5517 | 1.0000 | 0.9995 | 0.0355 | 0.5157 |
| Packet Length Mean | 1.0000 | 0.9947 | 1.0000 | 0.9943 | 0.0523 | 0.9168 |
| Packet Length Min | 0.0000 | 0.0000 | -0.0024 | 0.0000 | 0.0050 | 0.0000 |
| Packet Length Std | 1.0000 | 0.9979 | 1.0000 | 0.9947 | 0.0345 | 0.8975 |
| Packet Length Variance | 0.9995 | 0.9990 | 0.9995 | 0.9935 | 0.0391 | 0.9033 |
| RST Flag Count | 0.0000 | 0.5426 | 0.7927 | 0.9511 | 0.7191 | -0.0265 |
| SYN Flag Count | 0.0000 | 0.0005 | 0.0022 | 0.0719 | 0.6883 | 0.0000 |
| Subflow Bwd Packets | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Subflow Fwd Packets | 0.0000 | 0.0000 | -0.0008 | 0.0000 | 0.3772 | 0.0000 |
| Total Bwd packets | 0.8877 | 1.0000 | 0.5563 | 0.8472 | 0.7667 | -0.0167 |
| Total Fwd Packet | 0.1307 | 0.8862 | 0.9989 | 0.9365 | 0.7466 | 0.7587 |
| Total Length of Bwd Packet | 0.9995 | 0.9995 | 1.0000 | 0.9993 | 0.0389 | 0.9376 |
| Total Length of Fwd Packet | 0.9942 | 0.9884 | 0.7706 | 0.0258 | 0.0520 | 0.9351 |
| Total TCP Flow Time | 0.9827 | 1.0000 | 0.9016 | 0.6518 | 0.9525 | 0.8754 |

**Axis 2: separation stability per attack family** (71 features x 6 families)

| Feature | Botnet | BruteForce | DDoS | DoS | Infiltration | WebAttack |
|---|---:|---:|---:|---:|---:|---:|
| ACK Flag Count | 0.0599 | 0.7464 | 0.0596 | 0.0733 | 0.0004 | 0.8096 |
| Active Max | 0.0569 | 0.0569 | 0.0134 | 0.0386 | 0.0381 | 0.0240 |
| Active Mean | 0.0569 | 0.0569 | 0.0124 | 0.0386 | 0.0378 | 0.0237 |
| Active Min | 0.0569 | 0.0569 | 0.0123 | 0.0406 | 0.0382 | 0.0228 |
| Active Std | 0.0156 | 0.0156 | 0.0156 | 0.0101 | 0.0145 | 0.0156 |
| Bwd Bulk Rate Avg | 0.0066 | 0.0064 | 0.0079 | 0.0405 | 0.0066 | 0.0040 |
| Bwd Bytes/Bulk Avg | 0.0066 | 0.0064 | 0.0070 | 0.0369 | 0.0066 | 0.0040 |
| Bwd Header Length | 0.0723 | 0.8509 | 0.0623 | 0.2648 | 0.0332 | 0.8053 |
| Bwd IAT Max | 0.0120 | 0.0355 | 0.0296 | 0.0197 | 0.2900 | 0.2114 |
| Bwd IAT Mean | 0.0133 | 0.0011 | 0.0261 | 0.0154 | 0.2904 | 0.2330 |
| Bwd IAT Min | 0.0720 | 0.0046 | 0.0768 | 0.1095 | 0.2013 | 0.0847 |
| Bwd IAT Std | 0.0390 | 0.0768 | 0.0904 | 0.0657 | 0.1381 | 0.2740 |
| Bwd IAT Total | 0.0099 | 0.0834 | 0.0127 | 0.0196 | 0.2936 | 0.4370 |
| Bwd Init Win Bytes | 0.3113 | 0.0376 | 0.2954 | 0.2919 | 0.1289 | 0.5044 |
| Bwd PSH Flags | 0.1057 | 0.8930 | 0.0451 | 0.0418 | 0.1431 | 0.8124 |
| Bwd Packet Length Max | 0.0043 | -0.0868 | 0.3388 | 0.3144 | 0.7952 | 0.4412 |
| Bwd Packet Length Mean | 0.6035 | 0.0407 | 0.4829 | 0.3917 | 0.8106 | 0.8080 |
| Bwd Packet Length Min | 0.2632 | 0.2632 | 0.2632 | 0.2632 | 0.2588 | 0.2632 |
| Bwd Packet Length Std | 0.0319 | 0.1308 | 0.5557 | 0.4891 | 0.1512 | 0.2936 |
| Bwd Packets/s | 0.0975 | -0.0919 | -0.0852 | 0.0075 | 0.0817 | 0.2031 |
| Bwd RST Flags | 0.0051 | -0.0882 | 0.0051 | 0.0028 | 0.3989 | 0.0051 |
| Bwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CWR Flag Count | 0.0004 | 0.0002 | 0.0004 | 0.0002 | 0.0000 | 0.0004 |
| Down/Up Ratio | -0.1532 | 0.0122 | 0.0535 | 0.0149 | -0.0700 | 0.6398 |
| ECE Flag Count | 0.0003 | 0.0001 | 0.0003 | 0.0001 | 0.0000 | 0.0003 |
| FIN Flag Count | 0.5402 | 0.5399 | 0.3436 | 0.3772 | 0.0686 | 0.5350 |
| FWD Init Win Bytes | 0.2243 | 0.7625 | 0.4245 | 0.7417 | 0.0004 | 0.3608 |
| Flow Bytes/s | 0.0084 | -0.2002 | -0.0589 | 0.0840 | 0.8262 | 0.0014 |
| Flow Duration | 0.0230 | 0.0561 | 0.0337 | 0.0015 | 0.7491 | 0.4254 |
| Flow IAT Max | -0.0500 | 0.0060 | 0.0319 | 0.0041 | 0.7325 | 0.1985 |
| Flow IAT Mean | 0.1525 | -0.1301 | -0.1073 | 0.0199 | 0.7056 | 0.1689 |
| Flow IAT Min | -0.1053 | 0.0867 | 0.2039 | 0.2532 | 0.0025 | -0.0832 |
| Flow IAT Std | -0.0269 | 0.0323 | 0.0171 | 0.0041 | 0.3642 | 0.2002 |
| Flow Packets/s | 0.1030 | -0.1010 | -0.0836 | 0.0080 | 0.7168 | 0.1877 |
| Fwd Act Data Pkts | 0.0001 | 0.8604 | 0.0007 | 0.0005 | 0.4160 | 0.7435 |
| Fwd Bulk Rate Avg | 0.0000 | 0.0003 | 0.0002 | 0.0000 | 0.0000 | 0.0003 |
| Fwd Bytes/Bulk Avg | 0.0000 | 0.0003 | 0.0002 | 0.0000 | 0.0000 | 0.0003 |
| Fwd Header Length | 0.0454 | 0.7667 | 0.0533 | 0.2120 | -0.0298 | 0.8297 |
| Fwd IAT Max | -0.0612 | 0.0079 | 0.0342 | 0.0044 | 0.1607 | 0.1587 |
| Fwd IAT Mean | -0.0558 | -0.0741 | 0.0191 | 0.0069 | 0.1585 | 0.1965 |
| Fwd IAT Min | 0.1209 | 0.2102 | -0.0651 | -0.2198 | 0.0293 | 0.2408 |
| Fwd IAT Std | 0.0165 | 0.0142 | 0.1004 | 0.0266 | 0.1930 | 0.2681 |
| Fwd IAT Total | -0.0473 | 0.0659 | 0.0212 | 0.0127 | 0.1620 | 0.4989 |
| Fwd PSH Flags | 0.0206 | 0.9286 | 0.0203 | 0.0303 | 0.1458 | 0.8052 |
| Fwd Packet Length Max | 0.1840 | -0.0868 | -0.0912 | 0.2053 | 0.8151 | 0.5745 |
| Fwd Packet Length Mean | -0.0514 | -0.0732 | 0.0644 | 0.0829 | 0.8405 | 0.8688 |
| Fwd Packet Length Min | 0.2623 | 0.2623 | 0.2615 | 0.2600 | 0.2519 | 0.2623 |
| Fwd Packet Length Std | 0.2670 | 0.2145 | 0.1136 | 0.3092 | 0.1639 | 0.8111 |
| Fwd Packet/Bulk Avg | 0.0000 | 0.0003 | 0.0002 | 0.0000 | 0.0000 | 0.0003 |
| Fwd RST Flags | 0.0151 | 0.0151 | 0.0076 | -0.1809 | 0.0150 | 0.0146 |
| Fwd Seg Size Min | 0.1964 | 0.8369 | 0.1918 | 0.6335 | 0.6444 | 0.3444 |
| Fwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ICMP Code | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ICMP Type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Idle Mean | 0.0576 | 0.0575 | 0.0154 | 0.0333 | 0.0423 | 0.0261 |
| Idle Std | 0.0164 | 0.0164 | 0.0164 | 0.0101 | 0.0105 | 0.0164 |
| PSH Flag Count | 0.0723 | 0.9097 | 0.0292 | 0.0364 | 0.1483 | 0.8069 |
| Packet Length Max | 0.0704 | -0.0935 | 0.3302 | 0.3174 | 0.8222 | 0.4191 |
| Packet Length Mean | 0.4099 | -0.1007 | 0.2848 | 0.3123 | 0.8408 | 0.8052 |
| Packet Length Min | 0.2616 | 0.2616 | 0.2608 | 0.2616 | 0.2515 | 0.2616 |
| Packet Length Std | 0.0476 | -0.0495 | 0.4808 | 0.4385 | 0.7936 | 0.5626 |
| Packet Length Variance | 0.0476 | -0.0495 | 0.4808 | 0.4385 | 0.7934 | 0.5626 |
| RST Flag Count | 0.0334 | -0.1826 | -0.1384 | -0.3039 | 0.2522 | 0.0328 |
| SYN Flag Count | 0.3164 | 0.3164 | 0.3157 | 0.3427 | 0.0297 | 0.3164 |
| Subflow Bwd Packets | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Subflow Fwd Packets | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0098 | 0.0002 |
| Total Bwd packets | 0.0707 | 0.8004 | 0.0634 | 0.0751 | 0.3408 | 0.7927 |
| Total Fwd Packet | 0.0460 | 0.6810 | 0.0551 | 0.0695 | 0.1979 | 0.8159 |
| Total Length of Bwd Packet | 0.0919 | 0.1332 | 0.2756 | 0.2550 | 0.7977 | 0.8558 |
| Total Length of Fwd Packet | 0.0879 | 0.4825 | 0.0315 | 0.1075 | 0.8168 | 0.8605 |
| Total TCP Flow Time | 0.0234 | 0.0287 | 0.0310 | 0.0003 | 0.0240 | 0.3683 |

---
## Step 11: Cross-Analysis (C1–C9)

Step 11 joins feature importance (Step 5) with drift axes (Step 10) and runs C1 through C9 (several split into lettered sub-tests, e.g. C4a/C4b) characterizing how importance, covariate shift, concept stability, and cross-year transfer performance relate. C9 is [DECISIVE]: it retrains the real model on competing feature subsets and directly measures cross-year transfer F1.

### Reference: Naming Convention

One label scheme, no symbol reused across layers:

- **Hypotheses:** `H1` (8 independent tests, C4a-C7a/C4b-C7b, one per importance-variant x axis cell, no longer one combined two-axis verdict; BH-FDR corrected across exactly these 8), `H1.5` (4 supplementary delta-importance tests, C8a-C8d, Section 3, its own BH-FDR family), `H2` (the decisive ablation, C9).
- **Inputs** (measured quantities consumed by each analysis):
  | name | meaning | name | meaning |
  |------|---------|------|---------|
  | `imp_nat_2017` | native (gain) importance 2017 | `imp_nat_2018` | native (gain) importance 2018 |
  | `imp_perm_2017` | permutation importance 2017 | `imp_perm_2018` | permutation importance 2018 |
  | `cov_shift` | covariate (value) shift = **Axis 1** (calibrated C2ST) | `concept_stab` | separation stability = **Axis 2** |
  | `family` | attack family | | |
  | `benign_shift` | benign-only slice C2ST (calibrated) | `attack_shift` | attack-only slice C2ST (calibrated) |
  | `mi_2017` | mutual information 2017 | `mi_2018` | mutual information 2018 |
  | `rank_delta` | importance-rank change 2017→2018 | | |
- **Analyses:** `C1`–`C9`. `C9` is [DECISIVE].
- **Axes:** Axis 1 = `cov_shift`; Axis 2 = `concept_stab`.

### Dataset and model scope

- CIC-IDS 2017 and 2018, corrected 2022 re-extraction.
- Tree-ensemble NIDS (LightGBM-RF), importance-based detectors only.
- Not claimed: neural nets, SVMs, other ML families, other datasets.

---
## Section 1: Listings: feature and attack drift rankings

### C1: Listing: feature importance values (input: imp_nat_2017, imp_nat_2018, imp_perm_2017, imp_perm_2018)

**In short:** this is the same data, same 71 rows, as the "Feature Importance" table in Step 5 above (Nat-Bin-17/18, Nat-MC-17/18, Perm-Bin-17/18), not re-rendered a second time here. Permutation-MULTICLASS importance does not exist anywhere in this pipeline (confirmed: no `imp_perm_*_multi` column in `cross_table.csv`, and the Step 5 table never had one either), so this is 6 value columns, not 8.

_Source: Step 5 "Feature Importance" table above; CSVs in `output/5_training/`._



### C2a: Stability, binary framing, benign vs. pooled-attack (input: c2st_auc, c2st_auc_calibrated, separation_stability, benign_c2st, attack_c2st)

**In short:** two tables, feature-level benign-vs-pooled-attack readings in both axes (Table 1), and the existing flip/shift/collapse classification per feature (Table 2, reuses `df['verdict']`, no new computation).

**Table 1, C2ST-AUC (Axis 1) BEFORE and AFTER null calibration, benign-only vs attack-only:**

_Calibration: each slice gets its own NULL FLOOR, the C2ST-AUC you would still see if the two years were the SAME (pooled, randomly re-split, reclassified). Calibrated = (raw − null floor) / (1 − null floor), clipped to [0, 1]. 0 = indistinguishable from a same-year re-split; 1 = maximal separation. The POOLED C2ST-AUC (raw + calibrated) and the Wasserstein-shift benign/attack breakdown are in the Axis 1+2 summary table above (Step 10 walkthrough), not repeated here; this table adds the C2ST-AUC-SPECIFIC benign/attack breakdown (a different metric from that table's Wasserstein-shift columns), so you can see whether it was specifically C2ST-AUC, not just shift magnitude, that moved for benign vs attack traffic. There is no benign-only or attack-only variant of separation_stability, it is DEFINED as the benign-vs-attack gap, so it only exists as one pooled number per feature (see the summary table above for that column)._

| Feature | Benign C2ST (raw) | Benign null floor | Benign C2ST (calibrated) | Attack C2ST (raw) | Attack null floor | Attack C2ST (calibrated) |
|---|---:|---:|---:|---:|---:|---:|
| Bwd IAT Min | 0.8740 | 0.5190 | 0.7380 | 0.9508 | 0.5268 | 0.8959 |
| Flow IAT Max | 0.8413 | 0.5096 | 0.6764 | 0.9220 | 0.5185 | 0.8381 |
| Flow Duration | 0.8145 | 0.5324 | 0.6032 | 0.9143 | 0.5134 | 0.8239 |
| Bwd Packets/s | 0.8173 | 0.5190 | 0.6202 | 0.8719 | 0.5193 | 0.7334 |
| Flow Packets/s | 0.8291 | 0.5294 | 0.6368 | 0.8855 | 0.5238 | 0.7595 |
| Flow IAT Min | 0.8471 | 0.5165 | 0.6838 | 0.9154 | 0.5193 | 0.8240 |
| Fwd IAT Mean | 0.8097 | 0.5126 | 0.6095 | 0.8676 | 0.5129 | 0.7281 |
| Fwd IAT Min | 0.7564 | 0.5170 | 0.4956 | 0.9630 | 0.5215 | 0.9227 |
| Fwd IAT Max | 0.8188 | 0.5264 | 0.6175 | 0.9093 | 0.5208 | 0.8106 |
| Flow IAT Mean | 0.8034 | 0.5133 | 0.5961 | 0.8811 | 0.5053 | 0.7597 |
| Total Length of Fwd Packet | 0.8251 | 0.5238 | 0.6326 | 0.8418 | 0.5297 | 0.6637 |
| Packet Length Variance | 0.7197 | 0.5114 | 0.4263 | 0.9909 | 0.5116 | 0.9813 |
| Packet Length Max | 0.7114 | 0.5213 | 0.3971 | 0.9951 | 0.5143 | 0.9900 |
| Bwd Header Length | 0.7862 | 0.5223 | 0.5524 | 0.9393 | 0.5389 | 0.8683 |
| Bwd Packet Length Max | 0.7487 | 0.5258 | 0.4701 | 0.9915 | 0.5209 | 0.9823 |
| Bwd IAT Max | 0.8229 | 0.5168 | 0.6336 | 0.8804 | 0.5248 | 0.7483 |
| Total Length of Bwd Packet | 0.7682 | 0.5169 | 0.5202 | 0.9923 | 0.5302 | 0.9836 |
| Fwd IAT Total | 0.7968 | 0.5190 | 0.5776 | 0.8894 | 0.5290 | 0.7653 |
| Fwd Header Length | 0.7736 | 0.5147 | 0.5336 | 0.9819 | 0.5268 | 0.9617 |
| Packet Length Std | 0.7223 | 0.5214 | 0.4197 | 0.9916 | 0.5281 | 0.9822 |
| Bwd IAT Mean | 0.8225 | 0.5150 | 0.6339 | 0.8175 | 0.5172 | 0.6220 |
| FWD Init Win Bytes | 0.7337 | 0.5266 | 0.4375 | 0.9787 | 0.5293 | 0.9547 |
| Bwd Init Win Bytes | 0.6912 | 0.5231 | 0.3525 | 0.9923 | 0.5395 | 0.9833 |
| Bwd Packet Length Std | 0.6606 | 0.5156 | 0.2993 | 0.9901 | 0.5258 | 0.9792 |
| Flow Bytes/s | 0.7954 | 0.5188 | 0.5749 | 0.9029 | 0.5226 | 0.7967 |
| Total Bwd packets | 0.7470 | 0.5174 | 0.4758 | 0.9293 | 0.5336 | 0.8485 |
| Bwd IAT Total | 0.8005 | 0.5149 | 0.5888 | 0.8377 | 0.5192 | 0.6625 |
| Fwd Packet Length Max | 0.7221 | 0.5163 | 0.4255 | 0.8368 | 0.5216 | 0.6590 |
| Packet Length Mean | 0.6745 | 0.5330 | 0.3029 | 0.9922 | 0.5206 | 0.9838 |
| Flow IAT Std | 0.7588 | 0.5192 | 0.4982 | 0.9079 | 0.5326 | 0.8029 |
| Total Fwd Packet | 0.7408 | 0.5280 | 0.4508 | 0.9779 | 0.5233 | 0.9537 |
| Fwd Packet Length Std | 0.6680 | 0.5186 | 0.3103 | 0.9121 | 0.5230 | 0.8157 |
| Fwd Seg Size Min | 0.6942 | 0.5161 | 0.3679 | 0.7909 | 0.5134 | 0.5704 |
| Bwd Packet Length Mean | 0.6667 | 0.5201 | 0.3054 | 0.9922 | 0.5293 | 0.9834 |
| ACK Flag Count | 0.6543 | 0.5214 | 0.2777 | 0.9765 | 0.5237 | 0.9507 |
| Fwd Packet Length Mean | 0.6778 | 0.5090 | 0.3438 | 0.9545 | 0.5217 | 0.9050 |
| Fwd Act Data Pkts | 0.7108 | 0.5080 | 0.4121 | 0.7346 | 0.5237 | 0.4427 |
| Total TCP Flow Time | 0.6512 | 0.5200 | 0.2734 | 0.9024 | 0.5190 | 0.7972 |
| CWR Flag Count | 0.6442 | 0.5127 | 0.2700 | 0.7120 | 0.5143 | 0.4071 |
| Fwd IAT Std | 0.6936 | 0.5164 | 0.3663 | 0.9089 | 0.5125 | 0.8131 |
| Bwd IAT Std | 0.6595 | 0.5190 | 0.2921 | 0.8756 | 0.5191 | 0.7413 |
| ECE Flag Count | 0.6366 | 0.5166 | 0.2482 | 0.7117 | 0.5131 | 0.4080 |
| PSH Flag Count | 0.6421 | 0.5154 | 0.2615 | 0.7321 | 0.5297 | 0.4302 |
| Bwd PSH Flags | 0.6173 | 0.5258 | 0.1929 | 0.7355 | 0.5204 | 0.4485 |
| Down/Up Ratio | 0.6158 | 0.5152 | 0.2077 | 0.7864 | 0.5198 | 0.5552 |
| Fwd PSH Flags | 0.6186 | 0.5161 | 0.2119 | 0.7319 | 0.5222 | 0.4390 |
| RST Flag Count | 0.6443 | 0.5146 | 0.2672 | 0.9094 | 0.5226 | 0.8103 |
| Active Min | 0.6020 | 0.5175 | 0.1752 | 0.5571 | 0.5100 | 0.0963 |
| Bwd Packet Length Min | 0.6307 | 0.5135 | 0.2410 | 0.5000 | 0.5008 | -0.0017 |
| SYN Flag Count | 0.6012 | 0.5185 | 0.1719 | 0.6831 | 0.5163 | 0.3448 |
| Fwd RST Flags | 0.5958 | 0.5162 | 0.1646 | 0.7313 | 0.5186 | 0.4418 |
| Idle Mean | 0.5886 | 0.5160 | 0.1501 | 0.5698 | 0.5099 | 0.1222 |
| Fwd Packet Length Min | 0.6429 | 0.5153 | 0.2632 | 0.5002 | 0.5017 | -0.0029 |
| Bwd Bulk Rate Avg | 0.5401 | 0.5034 | 0.0740 | 0.6295 | 0.5159 | 0.2347 |
| Packet Length Min | 0.6242 | 0.5284 | 0.2032 | 0.5000 | 0.5008 | -0.0017 |
| Bwd Bytes/Bulk Avg | 0.5137 | 0.5058 | 0.0160 | 0.6310 | 0.5097 | 0.2473 |
| Active Max | 0.5472 | 0.5243 | 0.0481 | 0.5571 | 0.5108 | 0.0946 |
| Active Mean | 0.5594 | 0.5128 | 0.0957 | 0.5506 | 0.5120 | 0.0793 |
| Fwd Packet/Bulk Avg | 0.5222 | 0.5057 | 0.0335 | 0.5043 | 0.5018 | 0.0050 |
| Fwd Bytes/Bulk Avg | 0.5213 | 0.5064 | 0.0302 | 0.5100 | 0.5025 | 0.0151 |
| Bwd RST Flags | 0.5664 | 0.5146 | 0.1066 | 0.6853 | 0.5098 | 0.3579 |
| Fwd Bulk Rate Avg | 0.5243 | 0.5083 | 0.0326 | 0.5062 | 0.5026 | 0.0073 |
| ICMP Type | 0.5005 | 0.5017 | -0.0023 | 0.5000 | 0.5000 | 0.0000 |
| Subflow Bwd Packets | 0.5012 | 0.5000 | 0.0025 | 0.5000 | 0.5000 | 0.0000 |
| ICMP Code | 0.5005 | 0.5016 | -0.0022 | 0.5000 | 0.5000 | 0.0000 |
| Fwd URG Flags | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 |
| Bwd URG Flags | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 0.0000 |
| Subflow Fwd Packets | 0.5033 | 0.5064 | -0.0063 | 0.5302 | 0.5094 | 0.0426 |
| Active Std | 0.5093 | 0.5199 | -0.0220 | 0.5060 | 0.5034 | 0.0053 |
| FIN Flag Count | 0.5810 | 0.5238 | 0.1201 | 0.8654 | 0.5149 | 0.7226 |
| Idle Std | 0.4980 | 0.5090 | -0.0224 | 0.5040 | 0.5034 | 0.0012 |

**How the `verdict` column is actually decided** (a priority cascade, not a simple "C2ST-AUC value x separation_stability value" lookup, it runs on PER-YEAR separation-trust flags and direction signs, not the single combined columns shown above; this is the real rule from `10_execute_comparison.py`, checked in order, first match wins):

| Priority | Condition | Verdict |
|---:|---|---|
| 1 | Separation was trusted in BOTH 2017 and 2018, but the direction (benign vs. attack) reversed | flipped |
| 2 | Separation was trusted in 2017 but lost by 2018 | collapsed |
| 3 | Separation was never trusted in either year | weak |
| 4 | None of the above, the feature's distribution SHAPE changed between years (modality/structural route) AND its calibrated C2ST is above the threshold (the classifier confirms the mismatch is real, not GMM/type-detection noise) | restructured |
| 5 | None of the above, and calibrated C2ST-AUC > threshold (the two years are more classifier-distinguishable than this feature's own permutation-null noise) | shifted |
| 6 | None of the above (calibrated C2ST at or below the threshold) | stable |

_This 6-row table above is the actual `verdict` rule. Below is a second, simpler way to look at the same two axes: Axis 1 status (did the VALUE move) crossed with Axis 2 status (did the PATTERN survive), these two statuses, read together, ARE the combined picture; the single `verdict` column above is a priority cascade, not literally this combination._

**Table 2, per-feature status on both axes, plus the existing `verdict` cascade:**

| Feature | C2ST-AUC (raw) | C2ST-AUC (calibrated) | Axis 1 status | Separation stability | Axis 2 status | Verdict |
|---|---:|---:|---|---:|---|---|
| Bwd IAT Min | 0.8502 | 0.6857 | HIGH | 0.0819 | COLLAPSED | shifted |
| Flow IAT Max | 0.8279 | 0.6376 | HIGH | 0.2800 | WEAKENED | shifted |
| Flow Duration | 0.8228 | 0.6345 | HIGH | 0.3239 | WEAKENED | restructured |
| Bwd Packets/s | 0.8222 | 0.6260 | HIGH | 0.1809 | WEAKENED | restructured |
| Flow Packets/s | 0.8147 | 0.6147 | HIGH | 0.1790 | WEAKENED | restructured |
| Flow IAT Min | 0.8066 | 0.5995 | HIGH | 0.0909 | WEAKENED | shifted |
| Fwd IAT Mean | 0.7991 | 0.5833 | HIGH | 0.0856 | COLLAPSED | shifted |
| Fwd IAT Min | 0.7981 | 0.5811 | HIGH | -0.0926 | FLIPPED | flipped |
| Fwd IAT Max | 0.7923 | 0.5768 | HIGH | 0.1185 | WEAKENED | shifted |
| Flow IAT Mean | 0.7904 | 0.5703 | HIGH | 0.1708 | WEAKENED | restructured |
| Total Length of Fwd Packet | 0.7906 | 0.5533 | HIGH | 0.5944 | PRESERVED | shifted |
| Packet Length Variance | 0.7850 | 0.5492 | HIGH | 0.6981 | PRESERVED | shifted |
| Packet Length Max | 0.7848 | 0.5483 | HIGH | 0.7518 | PRESERVED | restructured |
| Bwd Header Length | 0.7834 | 0.5460 | HIGH | 0.6074 | PRESERVED | restructured |
| Bwd Packet Length Max | 0.7832 | 0.5435 | HIGH | 0.7165 | PRESERVED | restructured |
| Bwd IAT Max | 0.7768 | 0.5428 | HIGH | 0.1857 | WEAKENED | shifted |
| Total Length of Bwd Packet | 0.7772 | 0.5398 | HIGH | 0.7436 | PRESERVED | restructured |
| Fwd IAT Total | 0.7780 | 0.5383 | HIGH | 0.1100 | WEAKENED | shifted |
| Fwd Header Length | 0.7800 | 0.5315 | HIGH | 0.5874 | PRESERVED | shifted |
| Packet Length Std | 0.7679 | 0.5188 | HIGH | 0.6982 | PRESERVED | shifted |
| Bwd IAT Mean | 0.7560 | 0.4907 | HIGH | 0.1666 | WEAKENED | restructured |
| FWD Init Win Bytes | 0.7558 | 0.4863 | HIGH | 0.3956 | PRESERVED | shifted |
| Bwd Init Win Bytes | 0.7365 | 0.4602 | HIGH | 0.2337 | WEAKENED | restructured |
| Bwd Packet Length Std | 0.7308 | 0.4368 | MODERATE | 0.2557 | WEAKENED | shifted |
| Flow Bytes/s | 0.7319 | 0.4308 | MODERATE | 0.1912 | WEAKENED | restructured |
| Total Bwd packets | 0.7243 | 0.4235 | MODERATE | -0.1618 | FLIPPED | flipped |
| Bwd IAT Total | 0.7224 | 0.4190 | MODERATE | 0.1855 | WEAKENED | shifted |
| Fwd Packet Length Max | 0.7243 | 0.4167 | MODERATE | -0.6531 | FLIPPED | flipped |
| Packet Length Mean | 0.7169 | 0.4011 | MODERATE | 0.6744 | PRESERVED | restructured |
| Flow IAT Std | 0.7073 | 0.3960 | MODERATE | 0.1366 | WEAKENED | restructured |
| Total Fwd Packet | 0.7059 | 0.3950 | MODERATE | 0.1099 | WEAKENED | shifted |
| Fwd Packet Length Std | 0.6927 | 0.3715 | MODERATE | 0.1777 | WEAKENED | shifted |
| Fwd Seg Size Min | 0.7001 | 0.3705 | MODERATE | 0.4967 | PRESERVED | restructured |
| Bwd Packet Length Mean | 0.6969 | 0.3681 | MODERATE | 0.7574 | PRESERVED | shifted |
| ACK Flag Count | 0.6973 | 0.3611 | MODERATE | 0.3707 | PRESERVED | shifted |
| Fwd Packet Length Mean | 0.6940 | 0.3575 | MODERATE | 0.4674 | PRESERVED | shifted |
| Fwd Act Data Pkts | 0.6803 | 0.3353 | MODERATE | 0.1530 | WEAKENED | shifted |
| Total TCP Flow Time | 0.6777 | 0.3179 | MODERATE | 0.5719 | PRESERVED | shifted |
| CWR Flag Count | 0.6475 | 0.2698 | MODERATE | 0.0001 | COLLAPSED | restructured |
| Fwd IAT Std | 0.6513 | 0.2663 | MODERATE | 0.0601 | COLLAPSED | restructured |
| Bwd IAT Std | 0.6456 | 0.2584 | MODERATE | 0.1002 | WEAKENED | restructured |
| ECE Flag Count | 0.6405 | 0.2534 | MODERATE | 0.0000 | COLLAPSED | restructured |
| PSH Flag Count | 0.6319 | 0.2234 | MODERATE | 0.1144 | WEAKENED | shifted |
| Bwd PSH Flags | 0.6215 | 0.2179 | MODERATE | 0.1018 | WEAKENED | restructured |
| Down/Up Ratio | 0.6192 | 0.2071 | MODERATE | -0.0712 | FLIPPED | flipped |
| Fwd PSH Flags | 0.5939 | 0.1645 | LOW | 0.0991 | WEAKENED | shifted |
| RST Flag Count | 0.5943 | 0.1588 | LOW | -0.1053 | FLIPPED | flipped |
| Active Min | 0.5931 | 0.1547 | LOW | 0.0190 | COLLAPSED | shifted |
| Bwd Packet Length Min | 0.5873 | 0.1298 | LOW | 0.2623 | WEAKENED | shifted |
| SYN Flag Count | 0.5887 | 0.1213 | LOW | 0.2278 | WEAKENED | restructured |
| Fwd RST Flags | 0.5696 | 0.1163 | LOW | -0.0689 | FLIPPED | flipped |
| Idle Mean | 0.5705 | 0.1107 | LOW | 0.0191 | COLLAPSED | shifted |
| Fwd Packet Length Min | 0.5689 | 0.0963 | LOW | 0.2595 | WEAKENED | restructured |
| Bwd Bulk Rate Avg | 0.5580 | 0.0944 | LOW | 0.0083 | COLLAPSED | collapsed |
| Packet Length Min | 0.5688 | 0.0944 | LOW | 0.2594 | WEAKENED | restructured |
| Bwd Bytes/Bulk Avg | 0.5507 | 0.0806 | LOW | 0.0089 | COLLAPSED | collapsed |
| Active Max | 0.5385 | 0.0523 | LOW | 0.0143 | COLLAPSED | restructured |
| Active Mean | 0.5350 | 0.0313 | LOW | 0.0127 | COLLAPSED | shifted |
| Fwd Packet/Bulk Avg | 0.5212 | 0.0288 | LOW | 0.0002 | COLLAPSED | weak |
| Fwd Bytes/Bulk Avg | 0.5187 | 0.0269 | LOW | 0.0003 | COLLAPSED | weak |
| Bwd RST Flags | 0.5250 | 0.0254 | LOW | 0.0090 | COLLAPSED | collapsed |
| Fwd Bulk Rate Avg | 0.5167 | 0.0250 | LOW | 0.0004 | COLLAPSED | weak |
| ICMP Type | 0.5018 | 0.0035 | LOW | 0.0006 | COLLAPSED | weak |
| Subflow Bwd Packets | 0.5010 | 0.0003 | LOW | 0.0000 | COLLAPSED | weak |
| ICMP Code | 0.5000 | 0.0000 | NONE | 0.0006 | COLLAPSED | weak |
| Fwd URG Flags | 0.5000 | 0.0000 | NONE | 0.0000 | COLLAPSED | weak |
| Bwd URG Flags | 0.5000 | 0.0000 | NONE | 0.0000 | COLLAPSED | weak |
| Subflow Fwd Packets | 0.5087 | -0.0017 | NONE | 0.0005 | COLLAPSED | weak |
| Active Std | 0.5047 | -0.0061 | NONE | 0.0140 | COLLAPSED | stable |
| FIN Flag Count | 0.5144 | -0.0189 | NONE | 0.0702 | COLLAPSED | stable |
| Idle Std | 0.5046 | -0.0203 | NONE | 0.0137 | COLLAPSED | stable |

Axis 1 status (5 levels): bucketed directly from the CALIBRATED C2ST-AUC (see C2a Table 1 for the raw -> null floor -> calibrated derivation), NONE ≤0 | LOW 0-0.20 | MODERATE 0.20-0.45 | HIGH 0.45-0.70 | STRONG >0.70. Axis 2 status: PRESERVED (≥0.35, gap intact) / WEAKENED (0.09-0.35) / COLLAPSED (0-0.09) / FLIPPED (<0, direction reversed).

**Cross-tab, how many features land in each Axis-1 x Axis-2 combination:**

| Axis 1 \ Axis 2 | PRESERVED | WEAKENED | COLLAPSED | FLIPPED |
|---|---:|---:|---:|---:|
| NONE | 0 | 0 | 7 | 0 |
| LOW | 0 | 5 | 12 | 2 |
| MODERATE | 6 | 10 | 3 | 3 |
| HIGH | 9 | 11 | 2 | 1 |
| STRONG | 0 | 0 | 0 | 0 |

**What each combination means, and which features are actually in it this run:**

- **NONE x PRESERVED** (0 features this run):
  - Axis 1 (NONE): values did not move beyond what a same-year re-split would already show (at/below the null floor).
  - Axis 2 (PRESERVED): the attack-vs-benign gap held.
  - Meaning: Nothing moved and nothing broke. No action needed on this feature.
  - Features: none this run.

- **NONE x WEAKENED** (0 features this run):
  - Axis 1 (NONE): values did not move beyond what a same-year re-split would already show (at/below the null floor).
  - Axis 2 (WEAKENED): the gap got smaller but did not collapse.
  - Meaning: Values did not move, but the feature is losing its grip on the label. NOT a data-shift problem, retraining on fresher data will not explain this; the feature/label relationship itself needs re-examining.
  - Features: none this run.

- **NONE x COLLAPSED** (7 features this run):
  - Axis 1 (NONE): values did not move beyond what a same-year re-split would already show (at/below the null floor).
  - Axis 2 (COLLAPSED): the gap nearly disappeared.
  - Meaning: Values look the same, but the feature stopped separating attack from benign almost entirely. Most diagnostic failure mode: since the data did not move, simple retraining will not fix this, the feature needs re-engineering or dropping.
  - Features: Active Std, Bwd URG Flags, FIN Flag Count, Fwd URG Flags, ICMP Code, Idle Std, Subflow Fwd Packets

- **NONE x FLIPPED** (0 features this run):
  - Axis 1 (NONE): values did not move beyond what a same-year re-split would already show (at/below the null floor).
  - Axis 2 (FLIPPED): the gap reversed direction.
  - Meaning: Values did not shift, but the relationship reversed direction. Highest-priority concept drift: the model is being actively misled, not just less accurate. Needs re-engineering, not just retraining.
  - Features: none this run.

- **LOW x PRESERVED** (0 features this run):
  - Axis 1 (LOW): values moved slightly beyond the null floor, a small but real shift.
  - Axis 2 (PRESERVED): the attack-vs-benign gap held.
  - Meaning: The environment moved (a small amount) but this feature kept separating attack from benign through the move. Classic covariate shift with concept stability: a plain retrain on fresh data should restore performance.
  - Features: none this run.

- **LOW x WEAKENED** (5 features this run):
  - Axis 1 (LOW): values moved slightly beyond the null floor, a small but real shift.
  - Axis 2 (WEAKENED): the gap got smaller but did not collapse.
  - Meaning: Values moved (a small amount) AND discrimination weakened. Likely the dataset-wide shift dragging this feature down with it; retrain first, then re-check this feature.
  - Features: Bwd Packet Length Min, Fwd PSH Flags, Fwd Packet Length Min, Packet Length Min, SYN Flag Count

- **LOW x COLLAPSED** (12 features this run):
  - Axis 1 (LOW): values moved slightly beyond the null floor, a small but real shift.
  - Axis 2 (COLLAPSED): the gap nearly disappeared.
  - Meaning: Values moved (a small amount) and the feature lost almost all separating power, or a feature-specific failure that happens to coincide with it.
  - Features: Active Max, Active Mean, Active Min, Bwd Bulk Rate Avg, Bwd Bytes/Bulk Avg, Bwd RST Flags, Fwd Bulk Rate Avg, Fwd Bytes/Bulk Avg, Fwd Packet/Bulk Avg, ICMP Type, Idle Mean, Subflow Bwd Packets

- **LOW x FLIPPED** (2 features this run):
  - Axis 1 (LOW): values moved slightly beyond the null floor, a small but real shift.
  - Axis 2 (FLIPPED): the gap reversed direction.
  - Meaning: Values moved (a small amount) AND the relationship reversed, two compounding problems a retrain-and-forget pipeline would misread as ordinary drift.
  - Features: Fwd RST Flags, RST Flag Count

- **MODERATE x PRESERVED** (6 features this run):
  - Axis 1 (MODERATE): values moved a moderate amount beyond the null floor.
  - Axis 2 (PRESERVED): the attack-vs-benign gap held.
  - Meaning: The environment moved (a moderate amount) but this feature kept separating attack from benign through the move. Classic covariate shift with concept stability: a plain retrain on fresh data should restore performance.
  - Features: ACK Flag Count, Bwd Packet Length Mean, Fwd Packet Length Mean, Fwd Seg Size Min, Packet Length Mean, Total TCP Flow Time

- **MODERATE x WEAKENED** (10 features this run):
  - Axis 1 (MODERATE): values moved a moderate amount beyond the null floor.
  - Axis 2 (WEAKENED): the gap got smaller but did not collapse.
  - Meaning: Values moved (a moderate amount) AND discrimination weakened. Likely the dataset-wide shift dragging this feature down with it; retrain first, then re-check this feature.
  - Features: Bwd IAT Std, Bwd IAT Total, Bwd PSH Flags, Bwd Packet Length Std, Flow Bytes/s, Flow IAT Std, Fwd Act Data Pkts, Fwd Packet Length Std, PSH Flag Count, Total Fwd Packet

- **MODERATE x COLLAPSED** (3 features this run):
  - Axis 1 (MODERATE): values moved a moderate amount beyond the null floor.
  - Axis 2 (COLLAPSED): the gap nearly disappeared.
  - Meaning: Values moved (a moderate amount) and the feature lost almost all separating power, or a feature-specific failure that happens to coincide with it.
  - Features: CWR Flag Count, ECE Flag Count, Fwd IAT Std

- **MODERATE x FLIPPED** (3 features this run):
  - Axis 1 (MODERATE): values moved a moderate amount beyond the null floor.
  - Axis 2 (FLIPPED): the gap reversed direction.
  - Meaning: Values moved (a moderate amount) AND the relationship reversed, two compounding problems a retrain-and-forget pipeline would misread as ordinary drift.
  - Features: Down/Up Ratio, Fwd Packet Length Max, Total Bwd packets

- **HIGH x PRESERVED** (9 features this run):
  - Axis 1 (HIGH): values moved substantially beyond the null floor.
  - Axis 2 (PRESERVED): the attack-vs-benign gap held.
  - Meaning: The environment moved (a substantial amount) but this feature kept separating attack from benign through the move. Classic covariate shift with concept stability: a plain retrain on fresh data should restore performance, at this shift level, retraining is doing real, substantial work to keep up.
  - Features: Bwd Header Length, Bwd Packet Length Max, FWD Init Win Bytes, Fwd Header Length, Packet Length Max, Packet Length Std, Packet Length Variance, Total Length of Bwd Packet, Total Length of Fwd Packet

- **HIGH x WEAKENED** (11 features this run):
  - Axis 1 (HIGH): values moved substantially beyond the null floor.
  - Axis 2 (WEAKENED): the gap got smaller but did not collapse.
  - Meaning: Values moved (a substantial amount) AND discrimination weakened. Likely the dataset-wide shift dragging this feature down with it; retrain first, then re-check this feature.
  - Features: Bwd IAT Max, Bwd IAT Mean, Bwd Init Win Bytes, Bwd Packets/s, Flow Duration, Flow IAT Max, Flow IAT Mean, Flow IAT Min, Flow Packets/s, Fwd IAT Max, Fwd IAT Total

- **HIGH x COLLAPSED** (2 features this run):
  - Axis 1 (HIGH): values moved substantially beyond the null floor.
  - Axis 2 (COLLAPSED): the gap nearly disappeared.
  - Meaning: Values moved (a substantial amount) and the feature lost almost all separating power, at this shift level, dataset-wide drift is the more likely explanation.
  - Features: Bwd IAT Min, Fwd IAT Mean

- **HIGH x FLIPPED** (1 feature this run):
  - Axis 1 (HIGH): values moved substantially beyond the null floor.
  - Axis 2 (FLIPPED): the gap reversed direction.
  - Meaning: Values moved (a substantial amount) AND the relationship reversed, two compounding problems a retrain-and-forget pipeline would misread as ordinary drift.
  - Features: Fwd IAT Min

- **STRONG x PRESERVED** (0 features this run):
  - Axis 1 (STRONG): values moved almost to the point of being fully separable between years (calibrated C2ST-AUC near 1.0).
  - Axis 2 (PRESERVED): the attack-vs-benign gap held.
  - Meaning: The environment moved (a near-total amount) but this feature kept separating attack from benign through the move. Classic covariate shift with concept stability: a plain retrain on fresh data should restore performance, at this shift level, retraining is doing real, substantial work to keep up.
  - Features: none this run.

- **STRONG x WEAKENED** (0 features this run):
  - Axis 1 (STRONG): values moved almost to the point of being fully separable between years (calibrated C2ST-AUC near 1.0).
  - Axis 2 (WEAKENED): the gap got smaller but did not collapse.
  - Meaning: Values moved (a near-total amount) AND discrimination weakened. Likely the dataset-wide shift dragging this feature down with it; retrain first, then re-check this feature.
  - Features: none this run.

- **STRONG x COLLAPSED** (0 features this run):
  - Axis 1 (STRONG): values moved almost to the point of being fully separable between years (calibrated C2ST-AUC near 1.0).
  - Axis 2 (COLLAPSED): the gap nearly disappeared.
  - Meaning: Values moved (a near-total amount) and the feature lost almost all separating power, at this shift level, dataset-wide drift is the more likely explanation.
  - Features: none this run.

- **STRONG x FLIPPED** (0 features this run):
  - Axis 1 (STRONG): values moved almost to the point of being fully separable between years (calibrated C2ST-AUC near 1.0).
  - Axis 2 (FLIPPED): the gap reversed direction.
  - Meaning: Values moved (a near-total amount) AND the relationship reversed, two compounding problems a retrain-and-forget pipeline would misread as ordinary drift.
  - Features: none this run.




### C2b: Stability, multiclass framing, benign vs. each specific attack family (input: family)

Table A shows per-family C2ST-AUC (Axis 1), both RAW and CALIBRATED against that family's own null floor. Table B shows per-family separation stability (Axis 2).

**Table A, C2ST-AUC (Axis 1) per specific attack family, RAW:**

| Feature | Botnet | BruteForce | DDoS | DoS | Infiltration | WebAttack | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACK Flag Count | 0.9979 | 0.9995 | 0.9990 | 0.9579 | 0.8958 | 0.8958 | 0.6973 |
| Active Max | 0.5000 | 0.5005 | 0.7386 | 0.5342 | 0.5280 | 0.5190 | 0.5385 |
| Active Mean | 0.5000 | 0.5000 | 0.7323 | 0.5302 | 0.5316 | 0.5190 | 0.5350 |
| Active Min | 0.5000 | 0.5000 | 0.7314 | 0.5400 | 0.5368 | 0.5190 | 0.5931 |
| Active Std | 0.5000 | 0.5000 | 0.5000 | 0.5191 | 0.5045 | 0.5000 | 0.5047 |
| Bwd Bulk Rate Avg | 0.5000 | 0.5018 | 0.6028 | 0.8295 | 0.5000 | 0.5123 | 0.5580 |
| Bwd Bytes/Bulk Avg | 0.5000 | 0.5028 | 0.6025 | 0.8253 | 0.5000 | 0.5123 | 0.5507 |
| Bwd Header Length | 0.9460 | 0.9991 | 0.7907 | 0.9243 | 0.8980 | 0.9821 | 0.7834 |
| Bwd IAT Max | 0.9925 | 1.0000 | 0.8157 | 0.9172 | 0.7004 | 0.9206 | 0.7768 |
| Bwd IAT Mean | 0.9776 | 1.0000 | 0.8509 | 0.7597 | 0.7017 | 0.9671 | 0.7560 |
| Bwd IAT Min | 0.9894 | 0.9895 | 0.9876 | 0.9448 | 0.6929 | 0.9291 | 0.8502 |
| Bwd IAT Std | 0.9909 | 1.0000 | 0.8201 | 0.8835 | 0.5410 | 0.9337 | 0.6456 |
| Bwd IAT Total | 0.9920 | 1.0000 | 0.8211 | 0.8313 | 0.6922 | 0.9550 | 0.7224 |
| Bwd Init Win Bytes | 0.9965 | 0.9998 | 1.0000 | 0.9996 | 0.5186 | 0.9772 | 0.7365 |
| Bwd PSH Flags | 0.9524 | 0.7913 | 0.5250 | 0.5163 | 0.5267 | 0.4849 | 0.6215 |
| Bwd Packet Length Max | 1.0000 | 0.7888 | 1.0000 | 0.9994 | 0.5244 | 0.8112 | 0.7832 |
| Bwd Packet Length Mean | 0.9961 | 0.9995 | 1.0000 | 0.9992 | 0.5244 | 0.9269 | 0.6969 |
| Bwd Packet Length Min | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5025 | 0.5000 | 0.5873 |
| Bwd Packet Length Std | 0.9953 | 0.9995 | 1.0000 | 0.9994 | 0.5247 | 0.9506 | 0.7308 |
| Bwd Packets/s | 0.9740 | 1.0000 | 0.9533 | 0.7586 | 0.9798 | 0.9646 | 0.8222 |
| Bwd RST Flags | 0.5000 | 0.7790 | 0.5000 | 0.5050 | 0.8758 | 0.5000 | 0.5250 |
| Bwd URG Flags | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| CWR Flag Count | 1.0000 | 0.5000 | 0.9985 | 0.5000 | 0.5355 | 1.0000 | 0.6475 |
| Down/Up Ratio | 0.9986 | 0.9977 | 0.9934 | 0.8680 | 0.7195 | 0.9007 | 0.6192 |
| ECE Flag Count | 1.0000 | 0.5000 | 0.9990 | 0.5000 | 0.5330 | 1.0000 | 0.6405 |
| FIN Flag Count | 0.5000 | 0.5030 | 0.7540 | 0.8372 | 0.5278 | 0.5000 | 0.5144 |
| FWD Init Win Bytes | 0.5000 | 1.0000 | 0.8969 | 1.0000 | 0.5486 | 1.0000 | 0.7558 |
| Flow Bytes/s | 0.9918 | 1.0000 | 0.9695 | 0.9601 | 0.5305 | 0.9583 | 0.7319 |
| Flow Duration | 0.9932 | 1.0000 | 0.9504 | 0.8423 | 0.9772 | 0.9479 | 0.8228 |
| Flow IAT Max | 0.9933 | 1.0000 | 0.9495 | 0.9111 | 0.9797 | 0.9221 | 0.8279 |
| Flow IAT Mean | 0.9742 | 1.0000 | 0.9662 | 0.7986 | 0.9815 | 0.8927 | 0.7904 |
| Flow IAT Min | 0.9366 | 0.8383 | 0.7884 | 0.9782 | 0.9307 | 0.7359 | 0.8066 |
| Flow IAT Std | 0.9918 | 1.0000 | 0.9590 | 0.8905 | 0.7062 | 0.9092 | 0.7073 |
| Flow Packets/s | 0.9734 | 1.0000 | 0.9591 | 0.8013 | 0.9832 | 0.8877 | 0.8147 |
| Fwd Act Data Pkts | 0.5288 | 0.7913 | 0.5018 | 0.5119 | 0.5243 | 0.4600 | 0.6803 |
| Fwd Bulk Rate Avg | 0.5271 | 0.5005 | 0.5010 | 0.5149 | 0.5017 | 0.5000 | 0.5167 |
| Fwd Bytes/Bulk Avg | 0.5313 | 0.5015 | 0.5000 | 0.5144 | 0.5010 | 0.5000 | 0.5187 |
| Fwd Header Length | 0.5748 | 0.9479 | 1.0000 | 0.9608 | 0.8909 | 0.9804 | 0.7800 |
| Fwd IAT Max | 0.9953 | 1.0000 | 0.9544 | 0.9037 | 0.8872 | 0.9124 | 0.7923 |
| Fwd IAT Mean | 0.9657 | 1.0000 | 0.9674 | 0.8001 | 0.8886 | 0.9566 | 0.7991 |
| Fwd IAT Min | 0.8498 | 0.9108 | 0.9434 | 0.9857 | 0.8992 | 0.9479 | 0.7981 |
| Fwd IAT Std | 0.9921 | 1.0000 | 0.9621 | 0.8684 | 0.5387 | 0.9044 | 0.6513 |
| Fwd IAT Total | 0.9967 | 1.0000 | 0.9595 | 0.8497 | 0.8881 | 0.9455 | 0.7780 |
| Fwd PSH Flags | 0.5000 | 0.7960 | 0.5015 | 0.5003 | 0.5284 | 0.4600 | 0.5939 |
| Fwd Packet Length Max | 0.9998 | 0.7857 | 0.8917 | 0.5246 | 0.5277 | 0.9833 | 0.7243 |
| Fwd Packet Length Mean | 0.9998 | 0.9939 | 0.9998 | 0.9166 | 0.5294 | 0.9683 | 0.6940 |
| Fwd Packet Length Min | 0.5000 | 0.5000 | 0.5005 | 0.5028 | 0.5035 | 0.5000 | 0.5689 |
| Fwd Packet Length Std | 1.0000 | 0.9962 | 0.9998 | 0.8464 | 0.5295 | 0.9488 | 0.6927 |
| Fwd Packet/Bulk Avg | 0.5254 | 0.5010 | 0.5010 | 0.5158 | 0.5035 | 0.5000 | 0.5212 |
| Fwd RST Flags | 0.5000 | 0.5000 | 0.8832 | 0.9668 | 0.5008 | 0.5000 | 0.5696 |
| Fwd Seg Size Min | 0.5000 | 0.5000 | 0.5040 | 0.7458 | 0.5535 | 1.0000 | 0.7001 |
| Fwd URG Flags | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5030 | 0.5000 | 0.5000 |
| ICMP Code | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4990 | 0.5000 | 0.5000 |
| ICMP Type | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5015 | 0.5000 | 0.5018 |
| Idle Mean | 0.5000 | 0.5000 | 0.7840 | 0.5392 | 0.5267 | 0.5253 | 0.5705 |
| Idle Std | 0.5000 | 0.5000 | 0.5000 | 0.5197 | 0.5149 | 0.5000 | 0.5046 |
| PSH Flag Count | 0.9524 | 0.7993 | 0.5237 | 0.5153 | 0.5279 | 0.4849 | 0.6319 |
| Packet Length Max | 1.0000 | 0.7845 | 1.0000 | 0.9997 | 0.5247 | 0.7915 | 0.7848 |
| Packet Length Mean | 1.0000 | 0.9975 | 1.0000 | 0.9973 | 0.5323 | 0.9629 | 0.7169 |
| Packet Length Min | 0.5000 | 0.5000 | 0.5005 | 0.5000 | 0.5050 | 0.5000 | 0.5688 |
| Packet Length Std | 1.0000 | 0.9990 | 1.0000 | 0.9975 | 0.5269 | 0.9560 | 0.7679 |
| Packet Length Variance | 0.9998 | 0.9995 | 0.9998 | 0.9969 | 0.5258 | 0.9560 | 0.7850 |
| RST Flag Count | 0.5000 | 0.7835 | 0.8990 | 0.9769 | 0.8663 | 0.5000 | 0.5943 |
| SYN Flag Count | 0.5000 | 0.5002 | 0.5020 | 0.5438 | 0.8502 | 0.5000 | 0.5887 |
| Subflow Bwd Packets | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5010 |
| Subflow Fwd Packets | 0.5000 | 0.5000 | 0.5012 | 0.5000 | 0.7015 | 0.5000 | 0.5087 |
| Total Bwd packets | 0.9460 | 1.0000 | 0.7868 | 0.9276 | 0.8889 | 0.5250 | 0.7243 |
| Total Fwd Packet | 0.5754 | 0.9441 | 0.9995 | 0.9694 | 0.8778 | 0.9036 | 0.7059 |
| Total Length of Bwd Packet | 0.9998 | 0.9998 | 1.0000 | 0.9997 | 0.5266 | 0.9727 | 0.7772 |
| Total Length of Fwd Packet | 0.9972 | 0.9945 | 0.8892 | 0.5300 | 0.5325 | 0.9712 | 0.7906 |
| Total TCP Flow Time | 0.9918 | 1.0000 | 0.9523 | 0.8335 | 0.9772 | 0.9479 | 0.6777 |

_Source: `per_class_c2st_attribution.csv` (built from Step 10's `axis1_per_attack`)._

**Table A2, C2ST-AUC (Axis 1) per specific attack family, CALIBRATED:**

| Feature | Botnet | BruteForce | DDoS | DoS | Infiltration | WebAttack | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACK Flag Count | 0.9956 | 0.9989 | 0.9978 | 0.9119 | 0.7828 | 0.7159 | 0.3611 |
| Active Max | 0.0000 | 0.0010 | 0.4530 | 0.0526 | 0.0451 | -0.0404 | 0.0523 |
| Active Mean | -0.0017 | 0.0000 | 0.4461 | 0.0450 | 0.0484 | -0.0642 | 0.0313 |
| Active Min | 0.0000 | 0.0000 | 0.4430 | 0.0709 | 0.0605 | -0.0766 | 0.1547 |
| Active Std | 0.0000 | 0.0000 | 0.0000 | 0.0233 | 0.0021 | 0.0000 | -0.0061 |
| Bwd Bulk Rate Avg | 0.0000 | -0.0000 | 0.1714 | 0.6466 | 0.0000 | -0.0182 | 0.0944 |
| Bwd Bytes/Bulk Avg | 0.0000 | 0.0005 | 0.1891 | 0.6395 | 0.0000 | -0.0102 | 0.0806 |
| Bwd Header Length | 0.8887 | 0.9981 | 0.5527 | 0.8423 | 0.7849 | 0.9596 | 0.5460 |
| Bwd IAT Max | 0.9842 | 1.0000 | 0.6175 | 0.8295 | 0.3879 | 0.8201 | 0.5428 |
| Bwd IAT Mean | 0.9529 | 1.0000 | 0.6856 | 0.4981 | 0.3875 | 0.9237 | 0.4907 |
| Bwd IAT Min | 0.9773 | 0.9778 | 0.9741 | 0.8866 | 0.3677 | 0.8426 | 0.6857 |
| Bwd IAT Std | 0.9805 | 1.0000 | 0.6323 | 0.7574 | 0.0722 | 0.8498 | 0.2584 |
| Bwd IAT Total | 0.9828 | 1.0000 | 0.6279 | 0.6508 | 0.3645 | 0.8928 | 0.4190 |
| Bwd Init Win Bytes | 0.9929 | 0.9995 | 1.0000 | 0.9993 | 0.0300 | 0.9395 | 0.4602 |
| Bwd PSH Flags | 0.8983 | 0.5675 | 0.0344 | 0.0140 | 0.0358 | -0.2208 | 0.2179 |
| Bwd Packet Length Max | 1.0000 | 0.5519 | 1.0000 | 0.9988 | 0.0366 | 0.5702 | 0.5435 |
| Bwd Packet Length Mean | 0.9918 | 0.9990 | 1.0000 | 0.9983 | 0.0409 | 0.8371 | 0.3681 |
| Bwd Packet Length Min | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 | 0.0000 | 0.1298 |
| Bwd Packet Length Std | 0.9903 | 0.9990 | 1.0000 | 0.9988 | 0.0347 | 0.8878 | 0.4368 |
| Bwd Packets/s | 0.9458 | 1.0000 | 0.9044 | 0.5078 | 0.9581 | 0.9045 | 0.6260 |
| Bwd RST Flags | 0.0000 | 0.5349 | 0.0000 | 0.0053 | 0.7428 | 0.0000 | 0.0254 |
| Bwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CWR Flag Count | 1.0000 | 0.0000 | 0.9969 | 0.0000 | 0.0597 | 1.0000 | 0.2698 |
| Down/Up Ratio | 0.9970 | 0.9951 | 0.9862 | 0.7256 | 0.4202 | 0.7412 | 0.2071 |
| ECE Flag Count | 1.0000 | 0.0000 | 0.9979 | 0.0000 | 0.0548 | 1.0000 | 0.2534 |
| FIN Flag Count | 0.0000 | -0.0005 | 0.4856 | 0.6622 | 0.0395 | -0.0132 | -0.0189 |
| FWD Init Win Bytes | 0.0000 | 1.0000 | 0.7870 | 1.0000 | 0.0736 | 1.0000 | 0.4863 |
| Flow Bytes/s | 0.9826 | 1.0000 | 0.9376 | 0.9159 | 0.0468 | 0.8970 | 0.4308 |
| Flow Duration | 0.9859 | 1.0000 | 0.8950 | 0.6678 | 0.9523 | 0.8735 | 0.6345 |
| Flow IAT Max | 0.9859 | 1.0000 | 0.8933 | 0.8160 | 0.9571 | 0.8114 | 0.6376 |
| Flow IAT Mean | 0.9455 | 1.0000 | 0.9296 | 0.5820 | 0.9618 | 0.7562 | 0.5703 |
| Flow IAT Min | 0.8643 | 0.6627 | 0.5680 | 0.9535 | 0.8508 | 0.3655 | 0.5995 |
| Flow IAT Std | 0.9828 | 1.0000 | 0.9154 | 0.7739 | 0.3903 | 0.7725 | 0.3960 |
| Flow Packets/s | 0.9439 | 1.0000 | 0.9142 | 0.5922 | 0.9655 | 0.7012 | 0.6147 |
| Fwd Act Data Pkts | 0.0384 | 0.5654 | 0.0018 | 0.0023 | 0.0312 | -0.1513 | 0.3353 |
| Fwd Bulk Rate Avg | 0.0348 | -0.0024 | 0.0003 | 0.0135 | -0.0018 | 0.0000 | 0.0250 |
| Fwd Bytes/Bulk Avg | 0.0428 | -0.0005 | 0.0000 | 0.0173 | -0.0029 | 0.0000 | 0.0269 |
| Fwd Header Length | 0.1311 | 0.8913 | 1.0000 | 0.9184 | 0.7788 | 0.9557 | 0.5315 |
| Fwd IAT Max | 0.9902 | 1.0000 | 0.9055 | 0.8017 | 0.7671 | 0.7816 | 0.5768 |
| Fwd IAT Mean | 0.9266 | 1.0000 | 0.9309 | 0.5756 | 0.7736 | 0.9025 | 0.5833 |
| Fwd IAT Min | 0.6776 | 0.8169 | 0.8778 | 0.9703 | 0.7915 | 0.8742 | 0.5811 |
| Fwd IAT Std | 0.9834 | 1.0000 | 0.9217 | 0.7251 | 0.0719 | 0.7810 | 0.2663 |
| Fwd IAT Total | 0.9930 | 1.0000 | 0.9165 | 0.6905 | 0.7705 | 0.8645 | 0.5383 |
| Fwd PSH Flags | 0.0000 | 0.5764 | -0.0003 | -0.0249 | 0.0439 | -0.2137 | 0.1645 |
| Fwd Packet Length Max | 0.9995 | 0.5557 | 0.7750 | -0.0062 | 0.0280 | 0.9600 | 0.4167 |
| Fwd Packet Length Mean | 0.9995 | 0.9873 | 0.9995 | 0.8290 | 0.0525 | 0.9309 | 0.3575 |
| Fwd Packet Length Min | 0.0000 | 0.0000 | -0.0007 | 0.0018 | 0.0020 | 0.0000 | 0.0963 |
| Fwd Packet Length Std | 1.0000 | 0.9921 | 0.9995 | 0.6792 | 0.0469 | 0.8832 | 0.3715 |
| Fwd Packet/Bulk Avg | 0.0386 | 0.0003 | 0.0004 | 0.0182 | 0.0037 | 0.0000 | 0.0288 |
| Fwd RST Flags | 0.0000 | 0.0000 | 0.7576 | 0.9293 | -0.0002 | -0.0138 | 0.1163 |
| Fwd Seg Size Min | 0.0000 | -0.0017 | 0.0029 | 0.4643 | 0.0883 | 1.0000 | 0.3705 |
| Fwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0006 | 0.0000 | 0.0000 |
| ICMP Code | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0038 | 0.0000 | 0.0000 |
| ICMP Type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0003 | 0.0000 | 0.0035 |
| Idle Mean | 0.0000 | 0.0000 | 0.5563 | 0.0616 | 0.0375 | -0.0394 | 0.1107 |
| Idle Std | 0.0000 | 0.0000 | 0.0000 | 0.0246 | 0.0217 | 0.0000 | -0.0203 |
| PSH Flag Count | 0.9007 | 0.5774 | 0.0301 | 0.0089 | 0.0368 | -0.1767 | 0.2234 |
| Packet Length Max | 1.0000 | 0.5517 | 1.0000 | 0.9995 | 0.0355 | 0.5157 | 0.5483 |
| Packet Length Mean | 1.0000 | 0.9947 | 1.0000 | 0.9943 | 0.0523 | 0.9168 | 0.4011 |
| Packet Length Min | 0.0000 | 0.0000 | -0.0024 | 0.0000 | 0.0050 | 0.0000 | 0.0944 |
| Packet Length Std | 1.0000 | 0.9979 | 1.0000 | 0.9947 | 0.0345 | 0.8975 | 0.5188 |
| Packet Length Variance | 0.9995 | 0.9990 | 0.9995 | 0.9935 | 0.0391 | 0.9033 | 0.5492 |
| RST Flag Count | 0.0000 | 0.5426 | 0.7927 | 0.9511 | 0.7191 | -0.0265 | 0.1588 |
| SYN Flag Count | 0.0000 | 0.0005 | 0.0022 | 0.0719 | 0.6883 | 0.0000 | 0.1213 |
| Subflow Bwd Packets | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 |
| Subflow Fwd Packets | 0.0000 | 0.0000 | -0.0008 | 0.0000 | 0.3772 | 0.0000 | -0.0017 |
| Total Bwd packets | 0.8877 | 1.0000 | 0.5563 | 0.8472 | 0.7667 | -0.0167 | 0.4235 |
| Total Fwd Packet | 0.1307 | 0.8862 | 0.9989 | 0.9365 | 0.7466 | 0.7587 | 0.3950 |
| Total Length of Bwd Packet | 0.9995 | 0.9995 | 1.0000 | 0.9993 | 0.0389 | 0.9376 | 0.5398 |
| Total Length of Fwd Packet | 0.9942 | 0.9884 | 0.7706 | 0.0258 | 0.0520 | 0.9351 | 0.5533 |
| Total TCP Flow Time | 0.9827 | 1.0000 | 0.9016 | 0.6518 | 0.9525 | 0.8754 | 0.3179 |


**Table B, separation stability (Axis 2) per specific attack family:**

| Feature | Botnet | BruteForce | DDoS | DoS | Infiltration | WebAttack | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACK Flag Count | 0.0599 | 0.7464 | 0.0596 | 0.0733 | 0.0004 | 0.8096 | 0.3707 |
| Active Max | 0.0569 | 0.0569 | 0.0134 | 0.0386 | 0.0381 | 0.0240 | 0.0143 |
| Active Mean | 0.0569 | 0.0569 | 0.0124 | 0.0386 | 0.0378 | 0.0237 | 0.0127 |
| Active Min | 0.0569 | 0.0569 | 0.0123 | 0.0406 | 0.0382 | 0.0228 | 0.0190 |
| Active Std | 0.0156 | 0.0156 | 0.0156 | 0.0101 | 0.0145 | 0.0156 | 0.0140 |
| Bwd Bulk Rate Avg | 0.0066 | 0.0064 | 0.0079 | 0.0405 | 0.0066 | 0.0040 | 0.0083 |
| Bwd Bytes/Bulk Avg | 0.0066 | 0.0064 | 0.0070 | 0.0369 | 0.0066 | 0.0040 | 0.0089 |
| Bwd Header Length | 0.0723 | 0.8509 | 0.0623 | 0.2648 | 0.0332 | 0.8053 | 0.6074 |
| Bwd IAT Max | 0.0120 | 0.0355 | 0.0296 | 0.0197 | 0.2900 | 0.2114 | 0.1857 |
| Bwd IAT Mean | 0.0133 | 0.0011 | 0.0261 | 0.0154 | 0.2904 | 0.2330 | 0.1666 |
| Bwd IAT Min | 0.0720 | 0.0046 | 0.0768 | 0.1095 | 0.2013 | 0.0847 | 0.0819 |
| Bwd IAT Std | 0.0390 | 0.0768 | 0.0904 | 0.0657 | 0.1381 | 0.2740 | 0.1002 |
| Bwd IAT Total | 0.0099 | 0.0834 | 0.0127 | 0.0196 | 0.2936 | 0.4370 | 0.1855 |
| Bwd Init Win Bytes | 0.3113 | 0.0376 | 0.2954 | 0.2919 | 0.1289 | 0.5044 | 0.2337 |
| Bwd PSH Flags | 0.1057 | 0.8930 | 0.0451 | 0.0418 | 0.1431 | 0.8124 | 0.1018 |
| Bwd Packet Length Max | 0.0043 | -0.0868 | 0.3388 | 0.3144 | 0.7952 | 0.4412 | 0.7165 |
| Bwd Packet Length Mean | 0.6035 | 0.0407 | 0.4829 | 0.3917 | 0.8106 | 0.8080 | 0.7574 |
| Bwd Packet Length Min | 0.2632 | 0.2632 | 0.2632 | 0.2632 | 0.2588 | 0.2632 | 0.2623 |
| Bwd Packet Length Std | 0.0319 | 0.1308 | 0.5557 | 0.4891 | 0.1512 | 0.2936 | 0.2557 |
| Bwd Packets/s | 0.0975 | -0.0919 | -0.0852 | 0.0075 | 0.0817 | 0.2031 | 0.1809 |
| Bwd RST Flags | 0.0051 | -0.0882 | 0.0051 | 0.0028 | 0.3989 | 0.0051 | 0.0090 |
| Bwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CWR Flag Count | 0.0004 | 0.0002 | 0.0004 | 0.0002 | 0.0000 | 0.0004 | 0.0001 |
| Down/Up Ratio | -0.1532 | 0.0122 | 0.0535 | 0.0149 | -0.0700 | 0.6398 | -0.0712 |
| ECE Flag Count | 0.0003 | 0.0001 | 0.0003 | 0.0001 | 0.0000 | 0.0003 | 0.0000 |
| FIN Flag Count | 0.5402 | 0.5399 | 0.3436 | 0.3772 | 0.0686 | 0.5350 | 0.0702 |
| FWD Init Win Bytes | 0.2243 | 0.7625 | 0.4245 | 0.7417 | 0.0004 | 0.3608 | 0.3956 |
| Flow Bytes/s | 0.0084 | -0.2002 | -0.0589 | 0.0840 | 0.8262 | 0.0014 | 0.1912 |
| Flow Duration | 0.0230 | 0.0561 | 0.0337 | 0.0015 | 0.7491 | 0.4254 | 0.3239 |
| Flow IAT Max | -0.0500 | 0.0060 | 0.0319 | 0.0041 | 0.7325 | 0.1985 | 0.2800 |
| Flow IAT Mean | 0.1525 | -0.1301 | -0.1073 | 0.0199 | 0.7056 | 0.1689 | 0.1708 |
| Flow IAT Min | -0.1053 | 0.0867 | 0.2039 | 0.2532 | 0.0025 | -0.0832 | 0.0909 |
| Flow IAT Std | -0.0269 | 0.0323 | 0.0171 | 0.0041 | 0.3642 | 0.2002 | 0.1366 |
| Flow Packets/s | 0.1030 | -0.1010 | -0.0836 | 0.0080 | 0.7168 | 0.1877 | 0.1790 |
| Fwd Act Data Pkts | 0.0001 | 0.8604 | 0.0007 | 0.0005 | 0.4160 | 0.7435 | 0.1530 |
| Fwd Bulk Rate Avg | 0.0000 | 0.0003 | 0.0002 | 0.0000 | 0.0000 | 0.0003 | 0.0004 |
| Fwd Bytes/Bulk Avg | 0.0000 | 0.0003 | 0.0002 | 0.0000 | 0.0000 | 0.0003 | 0.0003 |
| Fwd Header Length | 0.0454 | 0.7667 | 0.0533 | 0.2120 | -0.0298 | 0.8297 | 0.5874 |
| Fwd IAT Max | -0.0612 | 0.0079 | 0.0342 | 0.0044 | 0.1607 | 0.1587 | 0.1185 |
| Fwd IAT Mean | -0.0558 | -0.0741 | 0.0191 | 0.0069 | 0.1585 | 0.1965 | 0.0856 |
| Fwd IAT Min | 0.1209 | 0.2102 | -0.0651 | -0.2198 | 0.0293 | 0.2408 | -0.0926 |
| Fwd IAT Std | 0.0165 | 0.0142 | 0.1004 | 0.0266 | 0.1930 | 0.2681 | 0.0601 |
| Fwd IAT Total | -0.0473 | 0.0659 | 0.0212 | 0.0127 | 0.1620 | 0.4989 | 0.1100 |
| Fwd PSH Flags | 0.0206 | 0.9286 | 0.0203 | 0.0303 | 0.1458 | 0.8052 | 0.0991 |
| Fwd Packet Length Max | 0.1840 | -0.0868 | -0.0912 | 0.2053 | 0.8151 | 0.5745 | -0.6531 |
| Fwd Packet Length Mean | -0.0514 | -0.0732 | 0.0644 | 0.0829 | 0.8405 | 0.8688 | 0.4674 |
| Fwd Packet Length Min | 0.2623 | 0.2623 | 0.2615 | 0.2600 | 0.2519 | 0.2623 | 0.2595 |
| Fwd Packet Length Std | 0.2670 | 0.2145 | 0.1136 | 0.3092 | 0.1639 | 0.8111 | 0.1777 |
| Fwd Packet/Bulk Avg | 0.0000 | 0.0003 | 0.0002 | 0.0000 | 0.0000 | 0.0003 | 0.0002 |
| Fwd RST Flags | 0.0151 | 0.0151 | 0.0076 | -0.1809 | 0.0150 | 0.0146 | -0.0689 |
| Fwd Seg Size Min | 0.1964 | 0.8369 | 0.1918 | 0.6335 | 0.6444 | 0.3444 | 0.4967 |
| Fwd URG Flags | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ICMP Code | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 |
| ICMP Type | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 |
| Idle Mean | 0.0576 | 0.0575 | 0.0154 | 0.0333 | 0.0423 | 0.0261 | 0.0191 |
| Idle Std | 0.0164 | 0.0164 | 0.0164 | 0.0101 | 0.0105 | 0.0164 | 0.0137 |
| PSH Flag Count | 0.0723 | 0.9097 | 0.0292 | 0.0364 | 0.1483 | 0.8069 | 0.1144 |
| Packet Length Max | 0.0704 | -0.0935 | 0.3302 | 0.3174 | 0.8222 | 0.4191 | 0.7518 |
| Packet Length Mean | 0.4099 | -0.1007 | 0.2848 | 0.3123 | 0.8408 | 0.8052 | 0.6744 |
| Packet Length Min | 0.2616 | 0.2616 | 0.2608 | 0.2616 | 0.2515 | 0.2616 | 0.2594 |
| Packet Length Std | 0.0476 | -0.0495 | 0.4808 | 0.4385 | 0.7936 | 0.5626 | 0.6982 |
| Packet Length Variance | 0.0476 | -0.0495 | 0.4808 | 0.4385 | 0.7934 | 0.5626 | 0.6981 |
| RST Flag Count | 0.0334 | -0.1826 | -0.1384 | -0.3039 | 0.2522 | 0.0328 | -0.1053 |
| SYN Flag Count | 0.3164 | 0.3164 | 0.3157 | 0.3427 | 0.0297 | 0.3164 | 0.2278 |
| Subflow Bwd Packets | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Subflow Fwd Packets | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0098 | 0.0002 | 0.0005 |
| Total Bwd packets | 0.0707 | 0.8004 | 0.0634 | 0.0751 | 0.3408 | 0.7927 | -0.1618 |
| Total Fwd Packet | 0.0460 | 0.6810 | 0.0551 | 0.0695 | 0.1979 | 0.8159 | 0.1099 |
| Total Length of Bwd Packet | 0.0919 | 0.1332 | 0.2756 | 0.2550 | 0.7977 | 0.8558 | 0.7436 |
| Total Length of Fwd Packet | 0.0879 | 0.4825 | 0.0315 | 0.1075 | 0.8168 | 0.8605 | 0.5944 |
| Total TCP Flow Time | 0.0234 | 0.0287 | 0.0310 | 0.0003 | 0.0240 | 0.3683 | 0.5719 |

**How to read these two tables together:**

Each row is one feature. The "Overall" column is that feature's pooled C2ST-AUC (Table A) or separation stability (Table B) across all attack traffic combined, the same number used everywhere else in this document. The per-family columns break that single pooled number down by which specific attack type it was tested against.

To find out which attack family DROVE a feature's overall shift or instability, compare the per-family columns against the Overall column for that row:
- If most per-family values are close to Overall, the shift/instability is broadly shared across attack types, no single family is responsible.
- If one or two families show a much higher (or much lower) value than the rest of the row, and those outlying values are close to Overall, that family is the one driving the feature's pooled result. The other families, sitting far from Overall, are NOT representative of why this feature shows up as shifted/unstable overall.
- Example pattern: a feature with Overall C2ST-AUC = 0.78, where 5 of 6 families sit near 0.50 (no shift) and one family sits at 0.80, that one family is responsible for the entire pooled shift signal; the feature did not generically drift, it drifted specifically against that attack type.

This is the same per-family data the family-stability-ranking analysis (Section 4's family instability check) draws on; this table is the feature-level view of the same underlying numbers.



### C3: Delta importance VALUE, 2018 minus 2017 (input: imp_nat_2017/2018, imp_perm_2017/2018)

**In short:** value change in importance across years, not positional/rank change, two tables, native and permutation, both sorted by |delta| descending.

**By native importance:**

| Feature | Native importance 2017 | Native importance 2018 | Delta (2018−2017) |
|---|---:|---:|---:|
| RST Flag Count | 0.3604 | 0.0003 | -0.3601 |
| FWD Init Win Bytes | 0.0566 | 0.3900 | +0.3334 |
| Bwd Init Win Bytes | 0.0146 | 0.2236 | +0.2090 |
| SYN Flag Count | 0.1222 | 0.0007 | -0.1214 |
| Bwd PSH Flags | 0.0856 | 0.0030 | -0.0825 |
| FIN Flag Count | 0.0055 | 0.0791 | +0.0737 |
| PSH Flag Count | 0.0424 | 0.0009 | -0.0415 |
| Bwd Packet Length Max | 0.0054 | 0.0379 | +0.0326 |
| Fwd Packet Length Mean | 0.0024 | 0.0346 | +0.0321 |
| Fwd Act Data Pkts | 0.0293 | 0.0029 | -0.0264 |
| Flow Duration | 0.0294 | 0.0058 | -0.0236 |
| Packet Length Max | 0.0013 | 0.0222 | +0.0208 |
| ACK Flag Count | 0.0211 | 0.0016 | -0.0195 |
| Packet Length Min | 0.0185 | 0.0000 | -0.0185 |
| Fwd IAT Min | 0.0196 | 0.0019 | -0.0176 |
| Bwd IAT Total | 0.0027 | 0.0198 | +0.0171 |
| Fwd Packet Length Min | 0.0124 | 0.0003 | -0.0121 |
| Bwd Packet Length Std | 0.0157 | 0.0277 | +0.0120 |
| Flow IAT Max | 0.0161 | 0.0052 | -0.0108 |
| Flow IAT Min | 0.0058 | 0.0165 | +0.0107 |
| Bwd IAT Max | 0.0126 | 0.0055 | -0.0071 |
| Bwd Packet Length Min | 0.0068 | 0.0000 | -0.0068 |
| Fwd PSH Flags | 0.0107 | 0.0044 | -0.0063 |
| Flow Bytes/s | 0.0006 | 0.0066 | +0.0060 |
| Bwd Packet Length Mean | 0.0049 | 0.0096 | +0.0047 |
| Flow IAT Mean | 0.0024 | 0.0063 | +0.0038 |
| Packet Length Variance | 0.0049 | 0.0012 | -0.0036 |
| Flow Packets/s | 0.0064 | 0.0028 | -0.0036 |
| Bwd Packets/s | 0.0016 | 0.0050 | +0.0034 |
| Total Length of Fwd Packet | 0.0055 | 0.0088 | +0.0032 |
| Total Length of Bwd Packet | 0.0047 | 0.0075 | +0.0028 |
| Fwd Packet Length Std | 0.0079 | 0.0106 | +0.0027 |
| Subflow Fwd Packets | 0.0026 | 0.0000 | -0.0026 |
| Packet Length Mean | 0.0014 | 0.0039 | +0.0025 |
| Bwd IAT Mean | 0.0031 | 0.0009 | -0.0022 |
| Bwd Header Length | 0.0009 | 0.0030 | +0.0021 |
| Fwd Header Length | 0.0016 | 0.0037 | +0.0021 |
| Packet Length Std | 0.0061 | 0.0039 | -0.0021 |
| Total Fwd Packet | 0.0023 | 0.0004 | -0.0020 |
| Bwd IAT Min | 0.0021 | 0.0003 | -0.0018 |
| Total Bwd packets | 0.0045 | 0.0033 | -0.0013 |
| Total TCP Flow Time | 0.0078 | 0.0067 | -0.0011 |
| Down/Up Ratio | 0.0026 | 0.0015 | -0.0011 |
| Idle Mean | 0.0011 | 0.0001 | -0.0010 |
| Fwd Packet Length Max | 0.0108 | 0.0117 | +0.0009 |
| Active Min | 0.0003 | 0.0011 | +0.0009 |
| CWR Flag Count | 0.0000 | 0.0008 | +0.0008 |
| ECE Flag Count | 0.0000 | 0.0005 | +0.0005 |
| Bwd IAT Std | 0.0006 | 0.0011 | +0.0005 |
| Bwd Bulk Rate Avg | 0.0005 | 0.0000 | -0.0005 |
| Bwd RST Flags | 0.0003 | 0.0007 | +0.0004 |
| Fwd Seg Size Min | 0.0105 | 0.0109 | +0.0004 |
| Fwd IAT Mean | 0.0006 | 0.0002 | -0.0004 |
| Fwd Bulk Rate Avg | 0.0005 | 0.0001 | -0.0004 |
| Active Mean | 0.0001 | 0.0004 | +0.0003 |
| Fwd IAT Std | 0.0005 | 0.0001 | -0.0003 |
| Fwd RST Flags | 0.0004 | 0.0001 | -0.0003 |
| Active Max | 0.0004 | 0.0001 | -0.0003 |
| Fwd IAT Total | 0.0004 | 0.0002 | -0.0003 |
| Fwd Packet/Bulk Avg | 0.0003 | 0.0006 | +0.0002 |
| Bwd Bytes/Bulk Avg | 0.0002 | 0.0000 | -0.0002 |
| Fwd IAT Max | 0.0003 | 0.0002 | -0.0002 |
| Idle Std | 0.0001 | 0.0000 | -0.0001 |
| Active Std | 0.0001 | 0.0000 | -0.0001 |
| ICMP Type | 0.0001 | 0.0000 | -0.0001 |
| Fwd URG Flags | 0.0000 | 0.0001 | +0.0001 |
| Flow IAT Std | 0.0004 | 0.0003 | -0.0001 |
| Fwd Bytes/Bulk Avg | 0.0005 | 0.0005 | -0.0000 |
| ICMP Code | 0.0001 | 0.0000 | -0.0000 |
| Subflow Bwd Packets | 0.0000 | 0.0000 | +0.0000 |
| Bwd URG Flags | 0.0000 | 0.0000 | +0.0000 |

**By permutation importance:**

| Feature | Permutation importance 2017 | Permutation importance 2018 | Delta (2018−2017) |
|---|---:|---:|---:|
| FWD Init Win Bytes | 0.0060 | 0.0926 | +0.0865 |
| FIN Flag Count | 0.0001 | 0.0333 | +0.0333 |
| RST Flag Count | 0.0055 | 0.0000 | -0.0055 |
| Bwd Init Win Bytes | 0.0011 | 0.0001 | -0.0010 |
| ECE Flag Count | 0.0000 | 0.0004 | +0.0004 |
| Fwd Seg Size Min | 0.0000 | 0.0004 | +0.0004 |
| Fwd PSH Flags | 0.0004 | -0.0000 | -0.0004 |
| ICMP Code | 0.0000 | 0.0003 | +0.0003 |
| CWR Flag Count | 0.0000 | 0.0002 | +0.0002 |
| SYN Flag Count | 0.0002 | 0.0000 | -0.0002 |
| Packet Length Max | 0.0000 | 0.0002 | +0.0001 |
| Bwd Packet Length Mean | 0.0000 | 0.0002 | +0.0001 |
| PSH Flag Count | 0.0001 | 0.0000 | -0.0001 |
| Bwd Packet Length Std | 0.0002 | 0.0001 | -0.0001 |
| Bwd Packet Length Max | 0.0000 | 0.0001 | +0.0001 |
| Bwd Packet Length Min | 0.0000 | 0.0001 | +0.0001 |
| Bwd PSH Flags | 0.0001 | 0.0000 | -0.0001 |
| Packet Length Min | 0.0000 | 0.0001 | +0.0001 |
| Fwd Packet Length Min | 0.0000 | 0.0001 | +0.0001 |
| Fwd Packet Length Max | 0.0001 | 0.0001 | +0.0001 |
| Packet Length Variance | 0.0000 | 0.0000 | -0.0000 |
| Fwd Packet Length Mean | 0.0000 | 0.0001 | +0.0000 |
| Fwd Header Length | -0.0000 | 0.0000 | +0.0000 |
| Total TCP Flow Time | -0.0000 | 0.0000 | +0.0000 |
| Fwd IAT Max | 0.0000 | 0.0000 | +0.0000 |
| Packet Length Mean | 0.0000 | 0.0000 | +0.0000 |
| Bwd IAT Min | 0.0000 | 0.0000 | +0.0000 |
| Active Min | 0.0000 | 0.0000 | +0.0000 |
| Bwd IAT Std | 0.0000 | 0.0000 | +0.0000 |
| Packet Length Std | 0.0001 | 0.0000 | -0.0000 |
| Fwd Bytes/Bulk Avg | 0.0000 | 0.0000 | +0.0000 |
| Flow IAT Max | -0.0000 | -0.0000 | +0.0000 |
| Total Length of Bwd Packet | 0.0000 | 0.0000 | +0.0000 |
| Bwd Header Length | 0.0000 | 0.0000 | +0.0000 |
| Fwd Bulk Rate Avg | 0.0000 | 0.0000 | +0.0000 |
| Bwd Packets/s | 0.0000 | 0.0000 | -0.0000 |
| Fwd IAT Min | 0.0000 | 0.0000 | -0.0000 |
| Flow Duration | -0.0000 | 0.0000 | +0.0000 |
| Fwd IAT Total | 0.0000 | 0.0000 | +0.0000 |
| Flow IAT Mean | -0.0000 | 0.0000 | +0.0000 |
| Bwd IAT Max | 0.0000 | 0.0000 | +0.0000 |
| Flow IAT Min | 0.0000 | 0.0000 | +0.0000 |
| Flow IAT Std | 0.0000 | 0.0000 | +0.0000 |
| Bwd IAT Mean | 0.0000 | 0.0000 | +0.0000 |
| Total Bwd packets | 0.0000 | 0.0000 | +0.0000 |
| Fwd Packet Length Std | 0.0000 | 0.0000 | +0.0000 |
| Flow Bytes/s | 0.0000 | 0.0000 | -0.0000 |
| Fwd Act Data Pkts | -0.0000 | 0.0000 | +0.0000 |
| Bwd Bulk Rate Avg | 0.0000 | 0.0000 | -0.0000 |
| Fwd RST Flags | 0.0000 | 0.0000 | +0.0000 |
| Flow Packets/s | -0.0000 | -0.0000 | +0.0000 |
| Fwd Packet/Bulk Avg | 0.0000 | 0.0000 | +0.0000 |
| ACK Flag Count | 0.0000 | -0.0000 | -0.0000 |
| Fwd IAT Mean | 0.0000 | 0.0000 | +0.0000 |
| Bwd IAT Total | 0.0000 | 0.0000 | -0.0000 |
| Active Max | 0.0000 | 0.0000 | +0.0000 |
| Fwd IAT Std | 0.0000 | 0.0000 | +0.0000 |
| Active Mean | 0.0000 | 0.0000 | +0.0000 |
| ICMP Type | -0.0000 | 0.0000 | +0.0000 |
| Total Length of Fwd Packet | 0.0000 | 0.0000 | -0.0000 |
| Total Fwd Packet | 0.0000 | 0.0000 | -0.0000 |
| Down/Up Ratio | 0.0000 | 0.0000 | -0.0000 |
| Bwd RST Flags | 0.0000 | -0.0000 | -0.0000 |
| Idle Mean | 0.0000 | 0.0000 | -0.0000 |
| Fwd URG Flags | 0.0000 | -0.0000 | -0.0000 |
| Bwd Bytes/Bulk Avg | 0.0000 | 0.0000 | +0.0000 |
| Active Std | 0.0000 | 0.0000 | +0.0000 |
| Bwd URG Flags | 0.0000 | 0.0000 | +0.0000 |
| Idle Std | 0.0000 | 0.0000 | +0.0000 |
| Subflow Fwd Packets | 0.0000 | 0.0000 | +0.0000 |
| Subflow Bwd Packets | 0.0000 | 0.0000 | +0.0000 |



---
## Section 2: H1, feature importance vs covariate shift AND concept stability

> **H1 claim, Axis 1 / C2ST-AUC:** High-importance features of a tree-ensemble NIDS show, on average, GREATER covariate shift (Axis 1, calibrated C2ST-AUC) between CIC-IDS 2017 and 2018 than low-importance features.
> **H1 claim, Axis 2 / separation_stability:** the SAME high-importance features PRESERVE the attack-vs-benign discrimination gap (Axis 2, separation_stability, already MI-aware) across years better than low-importance features. Both axes are tested with 4 importance variants each (native/permutation x 2017/2018), 8 independent tests total, each with its OWN verdict (no combined two-axis verdict). Below, each importance variant gets its Axis-1 cell (the "a" cell) immediately followed by its Axis-2 cell (the "b" cell): C4a/C4b, C5a/C5b, C6a/C6b, C7a/C7b.
> **Importance anchored to 2017 only:** The important/low feature sets are defined by 2017 importance so the shift measurement is strictly 2017→2018 and the result is not circular.
> Every cell below shows just the correlation/verdict, not a re-listing of all 71 features (those are already in C1/C2a/C2b above), and not the cluster-bootstrap or drift-vs-null checks either, which are side calculations not shown in the main test view since they would duplicate what the per-cell verdicts already summarize.

### C4a [HEADLINE]: Native importance (2017) vs covariate shift (Axis 1, calibrated C2ST-AUC)

We take native importance (2017), sort features high to low, and look at the covariate shift (Axis 1, calibrated C2ST-AUC) of those same features.

| Feature | native importance (2017) | covariate shift (C2ST-AUC) |
|---|---|---|
| _Features, ranked high-to-low by native importance (2017)_ | _raw importance value (Step 5)_ | _C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from._ |

**As native importance (2017) increases, covariate shift also increases**, a moderate positive correlation (ρ=+0.405).

Note:
1. Compared against Axis 1's C2ST-AUC score: C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI excludes zero: we are confident this is a real pattern, not just random noise from these specific features.



### C4b [HEADLINE]: Native importance (2017) vs concept stability (Axis 2, separation_stability)

Same native importance (2017) ranking as C4a, this time read against concept stability (Axis 2, separation_stability) of those same features.

| Feature | native importance (2017) | concept stability (separation_stability) |
|---|---|---|
| _Features, ranked high-to-low by native importance (2017)_ | _raw importance value (Step 5)_ | _separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read._ |

**As native importance (2017) increases, concept stability also increases**, a moderate positive correlation (ρ=+0.461).

Note:
1. Compared against Axis 2's separation_stability score: separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI excludes zero: we are confident this is a real pattern, not just random noise from these specific features.



### C5a: Native importance (2018) vs covariate shift (Axis 1, calibrated C2ST-AUC)

Same test as C4a, but native importance anchored to 2018 instead of 2017.

| Feature | native importance (2018) | covariate shift (C2ST-AUC) |
|---|---|---|
| _Features, ranked high-to-low by native importance (2018)_ | _raw importance value (Step 5)_ | _C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from._ |

**As native importance (2018) increases, covariate shift also increases**, a strong positive correlation (ρ=+0.593).

Note:
1. Compared against Axis 1's C2ST-AUC score: C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI excludes zero: we are confident this is a real pattern, not just random noise from these specific features.



### C5b: Native importance (2018) vs concept stability (Axis 2, separation_stability)

Same test as C4b, but native importance anchored to 2018 instead of 2017.

| Feature | native importance (2018) | concept stability (separation_stability) |
|---|---|---|
| _Features, ranked high-to-low by native importance (2018)_ | _raw importance value (Step 5)_ | _separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read._ |

**As native importance (2018) increases, concept stability also increases**, a strong positive correlation (ρ=+0.568).

Note:
1. Compared against Axis 2's separation_stability score: separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI excludes zero: we are confident this is a real pattern, not just random noise from these specific features.



### C6a: Permutation importance (2017) vs covariate shift (Axis 1, calibrated C2ST-AUC)

Same test as C4a, but using permutation importance instead of native importance. Native importance can be inflated for features with many distinct values; permutation importance is not, so this checks whether C4a's result depends on that bias.

| Feature | permutation importance (2017) | covariate shift (C2ST-AUC) |
|---|---|---|
| _Features, ranked high-to-low by permutation importance (2017)_ | _raw importance value (Step 5)_ | _C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from._ |

**permutation importance (2017) and covariate shift show no meaningful relationship**, essentially zero correlation (ρ=-0.006).

Note:
1. Compared against Axis 1's C2ST-AUC score: C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI includes zero: we cannot rule out that this is just random noise from these specific features, not a real pattern.

Comparing this to C4a: native and permutation importance point in DIFFERENT directions on this axis, so C4a's result may be driven by native importance's bias toward high-cardinality features, not a clean drift signal.



### C6b: Permutation importance (2017) vs concept stability (Axis 2, separation_stability)

Same test as C4b, but using permutation importance instead of native importance.

| Feature | permutation importance (2017) | concept stability (separation_stability) |
|---|---|---|
| _Features, ranked high-to-low by permutation importance (2017)_ | _raw importance value (Step 5)_ | _separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read._ |

**As permutation importance (2017) increases, concept stability also increases**, a weak positive correlation (ρ=+0.154).

Note:
1. Compared against Axis 2's separation_stability score: separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI includes zero: we cannot rule out that this is just random noise from these specific features, not a real pattern.

Comparing this to C4b: native and permutation importance point in DIFFERENT directions on this axis, so C4b's result may be driven by native importance's cardinality bias, not a clean concept-stability signal.



### C7a: Permutation importance (2018) vs covariate shift (Axis 1, calibrated C2ST-AUC)

Same test as C6a, but permutation importance anchored to 2018 instead of 2017.

| Feature | permutation importance (2018) | covariate shift (C2ST-AUC) |
|---|---|---|
| _Features, ranked high-to-low by permutation importance (2018)_ | _raw importance value (Step 5)_ | _C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from._ |

**As permutation importance (2018) increases, covariate shift also increases**, a weak positive correlation (ρ=+0.197).

Note:
1. Compared against Axis 1's C2ST-AUC score: C2ST-AUC, CALIBRATED against this feature's own null floor (0 = indistinguishable from a same-year re-split, 1 = fully separated). See C2a Table 1 for the raw value and null floor this was derived from. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI includes zero: we cannot rule out that this is just random noise from these specific features, not a real pattern.



### C7b: Permutation importance (2018) vs concept stability (Axis 2, separation_stability)

Same test as C6b, but permutation importance anchored to 2018 instead of 2017.

| Feature | permutation importance (2018) | concept stability (separation_stability) |
|---|---|---|
| _Features, ranked high-to-low by permutation importance (2018)_ | _raw importance value (Step 5)_ | _separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read._ |

**As permutation importance (2018) increases, concept stability also increases**, a moderate positive correlation (ρ=+0.340).

Note:
1. Compared against Axis 2's separation_stability score: separation_stability (near 1 = still separates attack/benign cleanly, near 0 = stopped, <0 = flipped). Raw value; the bucketed PRESERVED/WEAKENED/COLLAPSED/FLIPPED labels are a separate downstream read. (measured on the binary attack-vs-benign task, not the multiclass attack-TYPE task).
2. CI excludes zero: we are confident this is a real pattern, not just random noise from these specific features.



### H1 correlation summary: all 8 independent tests (binary importance, both axes)

**In short:** Axis 1 (covariate shift) is confirmed in 2/4 of the importance variants tested above; Axis 2 (concept stability) is contradicted in 3/4. So across these 8 correlation tests, H1 holds only partially and only on one axis, see the per-cell verdicts below for which variants agree, and C9 further down for the decisive (retrained-model) test, since these are supporting correlations, not the decisive evidence.

**Full detail:**

These are the same 8 cells computed in C4a-C7a (Axis 1) and C4b-C7b (Axis 2) above; this table is a rendering consolidation, not a new computation, so all 8 tests can be scanned side by side. Each cell carries its OWN independent verdict below, there is no single combined two-axis H1 verdict, since the two axes measure different things and a merged verdict would hide which one is driving a result. 95% CI is the cluster bootstrap where available (resampling collinearity clusters, the honest effective-n), else the plain bootstrap. `q=` is the Benjamini-Hochberg FDR-adjusted q-value computed across exactly these 8 tests (`*` = survives correction at alpha=0.05), running 8 correlated tests means some look "significant" by chance alone, and BH bounds the expected false-positive fraction among the starred cells.

**Multiple-testing correction (BH-FDR, alpha=0.05):** 5/8 of the H1 cells survive correction (uncorrected at p<0.05: 5). p-values: cluster bootstrap where available, else plain bootstrap.

| Importance variant | vs cov_shift (Axis 1) | vs concept_stab (Axis 2) |
|---------------------|------------------------|----------------------------|
| Native, 2017 (C4a / C4b) | +0.405 [+0.161,+0.608] q=0.008* | +0.461 [+0.228,+0.657] q=0.000* |
| Native, 2018 (C5a / C5b) | +0.593 [+0.373,+0.793] q=0.000* | +0.568 [+0.367,+0.715] q=0.000* |
| Permutation, 2017 (C6a / C6b) | -0.006 [-0.243,+0.296] q=0.954 | +0.154 [-0.093,+0.371] q=0.254 |
| Permutation, 2018 (C7a / C7b) | +0.197 [-0.076,+0.462] q=0.197 | +0.340 [+0.059,+0.580] q=0.024* |

**Same 8 results, in plain if/then terms:**
- If native importance (2017) is high, covariate shift tends to be high too (moderate correlation, ρ=+0.405).
- If native importance (2018) is high, covariate shift tends to be high too (strong correlation, ρ=+0.593).
- If permutation importance (2017) is high, covariate shift tends to be low too (weak correlation, ρ=-0.006).
- If permutation importance (2018) is high, covariate shift tends to be high too (weak correlation, ρ=+0.197).
- If native importance (2017) is high, concept stability tends to be high too (moderate correlation, ρ=+0.461).
- If native importance (2018) is high, concept stability tends to be high too (strong correlation, ρ=+0.568).
- If permutation importance (2017) is high, concept stability tends to be high too (weak correlation, ρ=+0.154).
- If permutation importance (2018) is high, concept stability tends to be high too (moderate correlation, ρ=+0.340).

H1 predicted covariate shift should go UP with importance (it does, in 2/4 of the tests above) and concept stability should go DOWN with importance (instead it goes UP in most of the tests above, the opposite of the prediction, contradicted in 3/4). So: the covariate-shift side of H1 holds, the concept-stability side does not.

Native vs permutation agreement, Axis 1: DISAGREE; Axis 2: DISAGREE. Agreement on an axis means that axis's native-importance result is not just a Gini-cardinality artifact (Strobl 2007); disagreement means lead with the permutation (unbiased) reading for that axis.

Multiclass importance (native or permutation) is not in this table: confirmed unused by any test in this pipeline; see the Step 5 importance listing above for the multiclass numbers themselves.

See Section 3 below for H1.5 (4 supplementary tests using DELTA importance, imp_2018 - imp_2017, instead of a year-anchored value), closer to H1 than to H2, no ablation companion.

---
## Section 3: H1.5, Delta importance vs stability (C8a-C8d, 4 tests)

We sort features by how much their importance VALUE changed between years (2018 value minus 2017 value, high-to-low), then ask: does a bigger change in importance go with a bigger change in stability? Tested against both axes, for BOTH the native (gain) and the permutation importance, all four computed cells are shown, not just the native pair, so the comparison is not selective. BH-FDR is applied within this 4-test family, separately from the 8 H1 cells.

### C8a: Delta native importance vs Axis 1

| Feature | Δ native importance | covariate shift (calibrated C2ST-AUC) |
|---|---|---|
| _Features, sorted by |2018−2017| importance delta_ | _raw delta value (Scripts 5+11)_ | _CALIBRATED C2ST-AUC (0 = at/below this feature's own null floor, 1 = fully separated between years)._ |

There is a weak positive correlation between how much native importance changed and covariate shift (ρ=+0.105), features whose importance changed the most also tend to have larger changes in covariate shift.

Note:
1. native importance delta = 2018 value minus 2017 value (Scripts 5 + 11)
2. Correlation with covariate shift (calibrated C2ST-AUC value, Script 10)
3. CI includes zero
4. No direction predicted in advance, unlike H1, not checked against an expected sign.
5. BH-FDR (within the 4-test H1.5 family): q=0.507, does NOT survive correction.



### C8b: Delta permutation importance vs Axis 1

| Feature | Δ permutation importance | covariate shift (calibrated C2ST-AUC) |
|---|---|---|
| _Features, sorted by |2018−2017| importance delta_ | _raw delta value (Scripts 5+11)_ | _CALIBRATED C2ST-AUC (0 = at/below this feature's own null floor, 1 = fully separated between years)._ |

There is a weak positive correlation between how much permutation importance changed and covariate shift (ρ=+0.077), features whose importance changed the most also tend to have larger changes in covariate shift.

Note:
1. permutation importance delta = 2018 value minus 2017 value (Scripts 5 + 11)
2. Correlation with covariate shift (calibrated C2ST-AUC value, Script 10)
3. CI includes zero
4. No direction predicted in advance, unlike H1, not checked against an expected sign.
5. BH-FDR (within the 4-test H1.5 family): q=0.508, does NOT survive correction.



### C8c: Delta native importance vs Axis 2

| Feature | Δ native importance | concept stability (separation_stability) |
|---|---|---|
| _Features, sorted by |2018−2017| importance delta_ | _raw delta value (Scripts 5+11)_ | _separation_stability (near 1 = still separates cleanly, near 0 = stopped, <0 = flipped). Raw value._ |

There is a weak positive correlation between how much native importance changed and concept stability (ρ=+0.176), features whose importance changed the most also tend to have larger changes in concept stability.

Note:
1. native importance delta = 2018 value minus 2017 value (Scripts 5 + 11)
2. Correlation with concept stability (separation_stability value, Script 10)
3. CI includes zero
4. No direction predicted in advance, unlike H1, not checked against an expected sign.
5. BH-FDR (within the 4-test H1.5 family): q=0.354, does NOT survive correction.



### C8d: Delta permutation importance vs Axis 2

| Feature | Δ permutation importance | concept stability (separation_stability) |
|---|---|---|
| _Features, sorted by |2018−2017| importance delta_ | _raw delta value (Scripts 5+11)_ | _separation_stability (near 1 = still separates cleanly, near 0 = stopped, <0 = flipped). Raw value._ |

There is a weak positive correlation between how much permutation importance changed and concept stability (ρ=+0.195), features whose importance changed the most also tend to have larger changes in concept stability.

Note:
1. permutation importance delta = 2018 value minus 2017 value (Scripts 5 + 11)
2. Correlation with concept stability (separation_stability value, Script 10)
3. CI includes zero
4. No direction predicted in advance, unlike H1, not checked against an expected sign.
5. BH-FDR (within the 4-test H1.5 family): q=0.354, does NOT survive correction.

_H1.5 is NOT an ablation input, no feature selection or retraining uses delta importance anywhere in this pipeline; it is a correlation-only side check._



---
## Section 4: Decisive experiment (C9)

> The correlations in Sections 2-3 are supporting evidence. The analysis below retrains / re-evaluates the real model and is the decisive test.

### C9 [DECISIVE]: Cross-domain ablation (H2 test)

**What we are doing, before any numbers:** we pick the top K features by a few different rules (K = 5, 10, 20, 30, 50, or all), train a real model on just those K features using 2017 data, and separately using 2018 data, then test each trained model on both years. This is the DECISIVE test for H2, it retrains the actual model and measures real transfer F1, not a correlation proxy like Sections 3-6 above.

**The selection rules compared (5 total, 3 we care about, plus a floor and a ceiling):**
  · `axis1_stable`, the K features with the LEAST covariate shift (Axis 1, C2ST-AUC; lower value = more stable). Same Axis-1 number as C2a's Table 1 ("overall" column) and the Step 10 axis table, looking at the OVERALL value for every row, not benign-only, attack-only, or any one specific attack.
  · `axis2_stable`, the K features with the MOST concept stability (Axis 2, separation_stability; higher value = more stable). Same overall-value convention.
  · `top_importance`, the K features the model relies on most (native importance, anchored to 2017 only, so feature SELECTION never peeks at 2018, the same list is reused for the 2018-trained models too).
  · `random`, K random features (floor reference, not a real competing policy).
  · `all_features`, all features, nothing dropped (ceiling reference).

We sort the full feature list by Axis-1 stability and separately by Axis-2 stability, take the top-K names off the TOP of each sorted list, then train BOTH a 2017 model and a 2018 model on just those K features. We do this for every K value above, for every policy above, that is a lot of trained models.

**How a trained model gets tested:** if a model trained on 2017 is tested on 2017 data, we scale with the 2017 scaler, no ambiguity. If a model trained on 2017 is tested on 2018 data, we test it TWICE: once scaling the 2018 test data with the 2018 scaler ("concept" framing, removes the location/scale shift, asks whether the decision boundary itself still works), and once scaling the 2018 test data with the 2017 scaler ("covariate" framing, the realistic deployment case, where a fresh scaler usually does not exist). Same thing in reverse for models trained on 2018. Every trained model ends up with both an in-domain F1 and these two cross-year F1 readings.

**What "H2 supported" would mean:** a stability-based policy (`axis1_stable` OR `axis2_stable`) would have to beat `top_importance` on cross-year F1 in BOTH directions (2017 model tested on 2018, AND 2018 model tested on 2017), both axes are tested here, head-to-head against `top_importance`, not axis 2 alone.

**Decision metric:** the verdict is decided on Macro F1 (mean of Attack F1 and Benign F1), not Attack F1 alone, Attack F1 ignores how a policy affects benign-traffic classification, and a policy that "wins" by flagging everything as an attack would look good on Attack F1 alone while wrecking benign traffic. Attack F1, Benign F1, sensitivity (attack recall), false-positive rate, precision, specificity, balanced accuracy, and MCC are all shown alongside as supporting context, none of the supporting metrics ever overrides the Macro F1 verdict.

  Output: `ablation_results.csv`, `ablation_crossdomain_f1.png`, `ablation_gap.png`
  Note: the test set is per-class capped (rebalanced), so the F1 numbers below are RELATIVE comparisons between policies, not natural-prior deployment numbers, see the real full-data baseline further down for that.

![Cross-domain macro F1 by policy and K](results/11_cross_analysis/lightgbm/ablation_crossdomain_f1.png)

![Generalization gap (in-domain minus cross-domain macro F1) by policy and K](results/11_cross_analysis/lightgbm/ablation_gap.png)

**Full H2 metric table, Attack F1 / Benign F1 / Macro F1 (decision metric) plus supporting context, cross-domain scenario, mean over every K:**

| Policy | Direction | Attack F1 | Benign F1 | Macro F1 | Sensitivity | FPR | Precision | Specificity | Balanced Acc | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top_importance | 2017->2018 | 0.623 | 0.699 | 0.661 | 0.584 | 0.080 | 0.853 | 0.920 | 0.752 | 0.492 |
| top_importance | 2018->2017 | 0.158 | 0.844 | 0.501 | 0.092 | 0.001 | 0.817 | 0.999 | 0.545 | 0.200 |
| axis1_stable | 2017->2018 | 0.048 | 0.468 | 0.258 | 0.031 | 0.026 | 0.338 | 0.974 | 0.502 | -0.017 |
| axis1_stable | 2018->2017 | 0.044 | 0.831 | 0.437 | 0.027 | 0.010 | 0.295 | 0.990 | 0.509 | 0.037 |
| axis2_stable | 2017->2018 | 0.388 | 0.523 | 0.455 | 0.294 | 0.112 | 0.720 | 0.888 | 0.591 | 0.161 |
| axis2_stable | 2018->2017 | 0.177 | 0.847 | 0.512 | 0.103 | 0.000 | 0.800 | 1.000 | 0.552 | 0.233 |

**Axis 1 / Axis 2 / top_importance, full breakdown by training year and K:**

Each table below is ONE selection policy, split into two stacked sub-tables, one for each year the model was TRAINED on. Within each sub-table the columns are the three train/test scenarios (in-domain, cross-year covariate framing, cross-year concept framing, see the method explanation above), each with its own F1 / Accuracy pair. The K=71 row in every sub-table is the SAME shared all-features ceiling, repeated everywhere so each sub-table is readable on its own, not three different measurements.

**Axis 1 (`axis1_stable`):**

_Trained on 2017:_

| K | F1 In-domain | MacroF1 In-domain | Acc In-domain | F1 Cross (covariate) | MacroF1 Cross (covariate) | Acc Cross (covariate) | F1 Cross (concept) | MacroF1 Cross (concept) | Acc Cross (concept) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.560 | 0.700 | 0.767 | 0.605 | 0.642 | 0.670 | 0.044 | 0.252 | 0.310 |
| 10 | 0.568 | 0.705 | 0.772 | 0.606 | 0.642 | 0.671 | 0.000 | 0.235 | 0.307 |
| 20 | 0.866 | 0.900 | 0.911 | 0.919 | 0.836 | 0.877 | 0.020 | 0.245 | 0.312 |
| 30 | 0.989 | 0.992 | 0.993 | 0.214 | 0.352 | 0.381 | 0.001 | 0.234 | 0.305 |
| 50 | 0.999 | 1.000 | 1.000 | 0.056 | 0.265 | 0.324 | 0.173 | 0.324 | 0.364 |
| 71 (all features) | 1.000 | 1.000 | 1.000 | 0.085 | 0.281 | 0.335 | 0.926 | 0.886 | 0.900 |

_Trained on 2018:_

| K | F1 In-domain | MacroF1 In-domain | Acc In-domain | F1 Cross (concept) | MacroF1 Cross (concept) | Acc Cross (concept) | F1 Cross (covariate) | MacroF1 Cross (covariate) | Acc Cross (covariate) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.925 | 0.882 | 0.898 | 0.042 | 0.429 | 0.694 | 0.218 | 0.482 | 0.618 |
| 10 | 0.925 | 0.882 | 0.898 | 0.000 | 0.416 | 0.712 | 0.223 | 0.487 | 0.623 |
| 20 | 0.966 | 0.943 | 0.952 | 0.010 | 0.420 | 0.710 | 0.285 | 0.547 | 0.699 |
| 30 | 0.991 | 0.986 | 0.988 | 0.000 | 0.415 | 0.709 | 0.522 | 0.697 | 0.797 |
| 50 | 1.000 | 1.000 | 1.000 | 0.166 | 0.505 | 0.739 | 0.533 | 0.708 | 0.813 |
| 71 (all features) | 1.000 | 1.000 | 1.000 | 0.258 | 0.556 | 0.755 | 0.608 | 0.752 | 0.835 |

**Axis 2 (`axis2_stable`):**

_Trained on 2017:_

| K | F1 In-domain | MacroF1 In-domain | Acc In-domain | F1 Cross (covariate) | MacroF1 Cross (covariate) | Acc Cross (covariate) | F1 Cross (concept) | MacroF1 Cross (concept) | Acc Cross (concept) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.941 | 0.958 | 0.964 | 0.059 | 0.255 | 0.307 | 0.140 | 0.282 | 0.311 |
| 10 | 0.993 | 0.995 | 0.996 | 0.056 | 0.263 | 0.321 | 0.124 | 0.293 | 0.334 |
| 20 | 0.999 | 1.000 | 1.000 | 0.057 | 0.266 | 0.325 | 0.619 | 0.593 | 0.595 |
| 30 | 1.000 | 1.000 | 1.000 | 0.140 | 0.313 | 0.357 | 0.624 | 0.594 | 0.596 |
| 50 | 1.000 | 1.000 | 1.000 | 0.088 | 0.281 | 0.333 | 0.433 | 0.514 | 0.546 |
| 71 (all features) | 1.000 | 1.000 | 1.000 | 0.085 | 0.281 | 0.335 | 0.926 | 0.886 | 0.900 |

_Trained on 2018:_

| K | F1 In-domain | MacroF1 In-domain | Acc In-domain | F1 Cross (concept) | MacroF1 Cross (concept) | Acc Cross (concept) | F1 Cross (covariate) | MacroF1 Cross (covariate) | Acc Cross (covariate) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.984 | 0.975 | 0.978 | 0.126 | 0.484 | 0.734 | 0.001 | 0.414 | 0.704 |
| 10 | 0.999 | 0.999 | 0.999 | 0.346 | 0.604 | 0.772 | 0.506 | 0.694 | 0.808 |
| 20 | 1.000 | 1.000 | 1.000 | 0.087 | 0.463 | 0.726 | 0.516 | 0.699 | 0.812 |
| 30 | 1.000 | 1.000 | 1.000 | 0.087 | 0.463 | 0.726 | 0.460 | 0.666 | 0.795 |
| 50 | 1.000 | 1.000 | 1.000 | 0.242 | 0.547 | 0.752 | 0.628 | 0.764 | 0.841 |
| 71 (all features) | 1.000 | 1.000 | 1.000 | 0.258 | 0.556 | 0.755 | 0.608 | 0.752 | 0.835 |

**top_importance (native, 2017-anchored, for comparison):**

_Trained on 2017:_

| K | F1 In-domain | MacroF1 In-domain | Acc In-domain | F1 Cross (covariate) | MacroF1 Cross (covariate) | Acc Cross (covariate) | F1 Cross (concept) | MacroF1 Cross (concept) | Acc Cross (concept) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.973 | 0.981 | 0.984 | 0.382 | 0.470 | 0.494 | 0.114 | 0.289 | 0.332 |
| 10 | 0.996 | 0.997 | 0.998 | 0.187 | 0.337 | 0.371 | 0.242 | 0.383 | 0.426 |
| 20 | 1.000 | 1.000 | 1.000 | 0.173 | 0.332 | 0.371 | 0.908 | 0.859 | 0.876 |
| 30 | 1.000 | 1.000 | 1.000 | 0.112 | 0.297 | 0.346 | 0.920 | 0.878 | 0.893 |
| 50 | 1.000 | 1.000 | 1.000 | 0.097 | 0.288 | 0.340 | 0.932 | 0.895 | 0.908 |
| 71 (all features) | 1.000 | 1.000 | 1.000 | 0.085 | 0.281 | 0.335 | 0.926 | 0.886 | 0.900 |

_Trained on 2018:_

| K | F1 In-domain | MacroF1 In-domain | Acc In-domain | F1 Cross (concept) | MacroF1 Cross (concept) | Acc Cross (concept) | F1 Cross (covariate) | MacroF1 Cross (covariate) | Acc Cross (covariate) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.986 | 0.977 | 0.981 | 0.277 | 0.565 | 0.757 | 0.517 | 0.695 | 0.799 |
| 10 | 0.999 | 0.998 | 0.999 | 0.000 | 0.416 | 0.712 | 0.402 | 0.632 | 0.775 |
| 20 | 1.000 | 0.999 | 1.000 | 0.009 | 0.420 | 0.712 | 0.713 | 0.815 | 0.870 |
| 30 | 1.000 | 1.000 | 1.000 | 0.254 | 0.554 | 0.754 | 0.635 | 0.768 | 0.844 |
| 50 | 1.000 | 1.000 | 1.000 | 0.250 | 0.551 | 0.752 | 0.592 | 0.742 | 0.829 |
| 71 (all features) | 1.000 | 1.000 | 1.000 | 0.258 | 0.556 | 0.755 | 0.608 | 0.752 | 0.835 |

**Results, per direction (read this; transfer is asymmetric):**

| Policy | Direction | Cross-domain Macro F1 | Gen. gap (Macro F1) |
|--------|-----------|----------------|---------|
| all_features | 2017->2018 | 0.886            | 0.113    |
| all_features | 2018->2017 | 0.556            | 0.444    |
| axis1_stable | 2017->2018 | 0.258            | 0.601    |
| axis1_stable | 2018->2017 | 0.437            | 0.502    |
| axis2_stable | 2017->2018 | 0.455            | 0.535    |
| axis2_stable | 2018->2017 | 0.512            | 0.482    |
| random | 2017->2018 | 0.424            | 0.571    |
| random | 2018->2017 | 0.437            | 0.560    |
| top_importance | 2017->2018 | 0.661            | 0.335    |
| top_importance | 2018->2017 | 0.501            | 0.494    |

**What we did and what we found:** we took the cross-domain Macro F1 (concept framing) for each policy, averaged it over every K value, but kept the two directions separate. Result: in 2017->2018, averaged over every K, top_importance=0.661, axis1_stable=0.258 (does not beat top_importance), axis2_stable=0.455 (does not beat top_importance); in 2018->2017, averaged over every K, top_importance=0.501, axis1_stable=0.437 (does not beat top_importance), axis2_stable=0.512 (beats top_importance). A policy has to win this comparison in BOTH directions at once to count as support for H2, winning only one direction is not enough; see the final verdict further down for that combined check.


**Results, mean over K and both directions (summary):**

| Policy | cross-Macro F1 (concept) | cross-Macro F1 (covariate) | gen. gap |
|--------|--------------------|---------------------|---------|
| all_features | 0.721              | 0.517              | 0.279    |
| axis1_stable | 0.348              | 0.566              | 0.551    |
| axis2_stable | 0.484              | 0.461              | 0.509    |
| random | 0.431              | 0.507              | 0.566    |
| top_importance | 0.581              | 0.538              | 0.414    |

**What we did and what we found (Macro F1):** same numbers as the table above, but now we also average the two directions together into one number per policy. Result: blending both directions together, axis1_stable averages 0.348 and axis2_stable averages 0.484 against top_importance's 0.581 (concept framing); on the covariate framing, axis1_stable=0.566, axis2_stable=0.461, top_importance=0.538. Because the two directions are blended into one number here, this table alone cannot show a policy that wins one direction and loses the other, that asymmetry only shows up in the per-direction table above and the full per-K tables below.

**Full per-K ablation results, 2017->2018** (every K actually run; each cell is the mean over the 5 seed replicates):

| Policy | K | Macro F1 in-domain | Macro F1 cross (concept) | Acc cross (concept) | Macro F1 cross (covariate) | Acc cross (covariate) | F1 in-domain (swap control) | Gen. gap (Macro F1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_features | 71 | 1.000 | 0.886 | 0.900 | 0.281 | 0.335 | 0.432 | 0.113 |
| axis1_stable | 5 | 0.700 | 0.252 | 0.310 | 0.642 | 0.670 | 0.216 | 0.448 |
| axis1_stable | 10 | 0.705 | 0.235 | 0.307 | 0.642 | 0.671 | 0.000 | 0.471 |
| axis1_stable | 20 | 0.900 | 0.245 | 0.312 | 0.836 | 0.877 | 0.018 | 0.655 |
| axis1_stable | 30 | 0.992 | 0.234 | 0.305 | 0.352 | 0.381 | 0.150 | 0.757 |
| axis1_stable | 50 | 1.000 | 0.324 | 0.364 | 0.265 | 0.324 | 0.847 | 0.675 |
| axis2_stable | 5 | 0.958 | 0.282 | 0.311 | 0.255 | 0.307 | 0.790 | 0.675 |
| axis2_stable | 10 | 0.995 | 0.293 | 0.334 | 0.263 | 0.321 | 0.849 | 0.701 |
| axis2_stable | 20 | 1.000 | 0.593 | 0.595 | 0.266 | 0.325 | 0.922 | 0.407 |
| axis2_stable | 30 | 1.000 | 0.594 | 0.596 | 0.313 | 0.357 | 0.922 | 0.406 |
| axis2_stable | 50 | 1.000 | 0.514 | 0.546 | 0.281 | 0.333 | 0.541 | 0.485 |
| random | 5 | 0.983 | 0.445 | 0.480 | 0.515 | 0.537 | 0.456 | 0.538 |
| random | 10 | 0.996 | 0.243 | 0.308 | 0.281 | 0.334 | 0.531 | 0.753 |
| random | 20 | 0.999 | 0.325 | 0.373 | 0.325 | 0.370 | 0.479 | 0.674 |
| random | 30 | 0.999 | 0.233 | 0.292 | 0.263 | 0.323 | 0.377 | 0.766 |
| random | 50 | 1.000 | 0.875 | 0.889 | 0.276 | 0.331 | 0.545 | 0.124 |
| top_importance | 5 | 0.981 | 0.289 | 0.332 | 0.470 | 0.494 | 0.740 | 0.692 |
| top_importance | 10 | 0.997 | 0.383 | 0.426 | 0.337 | 0.371 | 0.069 | 0.614 |
| top_importance | 20 | 1.000 | 0.859 | 0.876 | 0.332 | 0.371 | 0.645 | 0.141 |
| top_importance | 30 | 1.000 | 0.878 | 0.893 | 0.297 | 0.346 | 0.480 | 0.121 |
| top_importance | 50 | 1.000 | 0.895 | 0.908 | 0.288 | 0.340 | 0.347 | 0.105 |

**What we did and what we found, direction 2017->2018:** we compared every K value one at a time on cross-domain F1 (concept framing), instead of averaging across K like the two tables above. Result: axis1_stable beats top_importance at K = none of the tested K values; axis2_stable beats top_importance at K = [5]. H2 needs a policy to win essentially every K, in both directions, to be a reliable effect rather than a lucky K choice.

**Full per-K ablation results, 2018->2017** (every K actually run; each cell is the mean over the 5 seed replicates):

| Policy | K | Macro F1 in-domain | Macro F1 cross (concept) | Acc cross (concept) | Macro F1 cross (covariate) | Acc cross (covariate) | F1 in-domain (swap control) | Gen. gap (Macro F1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_features | 71 | 1.000 | 0.556 | 0.755 | 0.752 | 0.835 | 0.060 | 0.444 |
| axis1_stable | 5 | 0.882 | 0.429 | 0.694 | 0.482 | 0.618 | 0.184 | 0.453 |
| axis1_stable | 10 | 0.882 | 0.416 | 0.712 | 0.487 | 0.623 | 0.000 | 0.466 |
| axis1_stable | 20 | 0.943 | 0.420 | 0.710 | 0.547 | 0.699 | 0.011 | 0.523 |
| axis1_stable | 30 | 0.986 | 0.415 | 0.709 | 0.697 | 0.797 | 0.002 | 0.571 |
| axis1_stable | 50 | 1.000 | 0.505 | 0.739 | 0.708 | 0.813 | 0.022 | 0.494 |
| axis2_stable | 5 | 0.975 | 0.484 | 0.734 | 0.414 | 0.704 | 0.000 | 0.490 |
| axis2_stable | 10 | 0.999 | 0.604 | 0.772 | 0.694 | 0.808 | 0.001 | 0.395 |
| axis2_stable | 20 | 1.000 | 0.463 | 0.726 | 0.699 | 0.812 | 0.000 | 0.537 |
| axis2_stable | 30 | 1.000 | 0.463 | 0.726 | 0.666 | 0.795 | 0.000 | 0.537 |
| axis2_stable | 50 | 1.000 | 0.547 | 0.752 | 0.764 | 0.841 | 0.001 | 0.453 |
| random | 5 | 0.993 | 0.468 | 0.722 | 0.769 | 0.844 | 0.013 | 0.524 |
| random | 10 | 0.997 | 0.426 | 0.715 | 0.641 | 0.785 | 0.000 | 0.570 |
| random | 20 | 0.999 | 0.416 | 0.712 | 0.663 | 0.795 | 0.000 | 0.583 |
| random | 30 | 0.998 | 0.431 | 0.716 | 0.667 | 0.797 | 0.001 | 0.567 |
| random | 50 | 1.000 | 0.444 | 0.721 | 0.668 | 0.797 | 0.007 | 0.556 |
| top_importance | 5 | 0.977 | 0.565 | 0.757 | 0.695 | 0.799 | 0.002 | 0.412 |
| top_importance | 10 | 0.998 | 0.416 | 0.712 | 0.632 | 0.775 | 0.000 | 0.582 |
| top_importance | 20 | 0.999 | 0.420 | 0.712 | 0.815 | 0.870 | 0.002 | 0.579 |
| top_importance | 30 | 1.000 | 0.554 | 0.754 | 0.768 | 0.844 | 0.001 | 0.446 |
| top_importance | 50 | 1.000 | 0.551 | 0.752 | 0.742 | 0.829 | 0.036 | 0.448 |

**What we did and what we found, direction 2018->2017:** we compared every K value one at a time on cross-domain F1 (concept framing), instead of averaging across K like the two tables above. Result: axis1_stable beats top_importance at K = [20]; axis2_stable beats top_importance at K = [10, 20]. H2 needs a policy to win essentially every K, in both directions, to be a reliable effect rather than a lucky K choice.

**Reading the table above, down each column (one policy, K rising) and across each row (one K, every policy), tracked on Macro F1:**

_Direction 2017->2018, scanning down, does Macro F1 rise as K rises?_

  `top_importance`: rises as K increases; peak Macro F1=0.895 at K=50 (boundary of the tested range); biggest single step K=10→20 (0.383→0.859).
  `axis1_stable`: is non-monotonic as K increases; peak Macro F1=0.324 at K=50 (boundary of the tested range); biggest single step K=30→50 (0.234→0.324).
  `axis2_stable`: is non-monotonic as K increases; peak Macro F1=0.594 at K=30 (an interior peak); biggest single step K=10→20 (0.293→0.593).
  `random`: is non-monotonic as K increases; peak Macro F1=0.875 at K=50 (boundary of the tested range); biggest single step K=30→50 (0.233→0.875).

_Direction 2017->2018, scanning across, best policy at each K:_

  K=5: best is `random` (Macro F1=0.445)
  K=10: best is `top_importance` (Macro F1=0.383)
  K=20: best is `top_importance` (Macro F1=0.859)
  K=30: best is `top_importance` (Macro F1=0.878)
  K=50: best is `top_importance` (Macro F1=0.895)

_Direction 2018->2017, scanning down, does Macro F1 rise as K rises?_

  `top_importance`: is non-monotonic as K increases; peak Macro F1=0.565 at K=5 (boundary of the tested range); biggest single step K=5→10 (0.565→0.416).
  `axis1_stable`: is non-monotonic as K increases; peak Macro F1=0.505 at K=50 (boundary of the tested range); biggest single step K=30→50 (0.415→0.505).
  `axis2_stable`: is non-monotonic as K increases; peak Macro F1=0.604 at K=10 (an interior peak); biggest single step K=10→20 (0.604→0.463).
  `random`: is non-monotonic as K increases; peak Macro F1=0.468 at K=5 (boundary of the tested range); biggest single step K=5→10 (0.468→0.426).

_Direction 2018->2017, scanning across, best policy at each K:_

  K=5: best is `top_importance` (Macro F1=0.565)
  K=10: best is `axis2_stable` (Macro F1=0.604)
  K=20: best is `axis2_stable` (Macro F1=0.463)
  K=30: best is `top_importance` (Macro F1=0.554)
  K=50: best is `top_importance` (Macro F1=0.551)

`all_features` is left out of this scan since it only has one K (71, the ceiling reference), see the Effect-size table below for how K=71 compares to small K.

**What this means for H2:** counting every (K, direction) combination shown above, `top_importance` wins 7/10 (K, direction) combinations, `axis2_stable` wins 3/10 (K, direction) combinations. H2 needs one of the two stability axes to win essentially all of them, not just a plurality.

To keep in mind: `random` and `all_features` are the floor/ceiling references, not competing policies for H2; `all_features` only runs at K=71 (all features, nothing dropped), the other four policies run at every K. The swap-control column above is the SAME in-domain rows as the F1 in-domain column, just re-expressed in the other year's scaler, a large drop there (with no real distribution shift) would mean the model is sensitive to scaler mismatch itself, not just to genuine cross-year drift; a small drop means the COVARIATE column's degradation is mostly real shift, not a scaling artifact.

**Effect size (K=5 vs K=71, "median F1 of top-K vs bottom-K" reading for the ablation):** since each policy always keeps its TOP-K features (there is no natural bottom-K group in the ablation framework), the closest analogous contrast is the smallest vs largest feature subset actually tested:

| Policy | Direction | Cross-F1 at K=5 | Cross-F1 at K=71 (all_features) | Gain |
|---|---|---:|---:|---:|
| top_importance | 2017->2018 | 0.114 | 0.926 | +0.812 |
| top_importance | 2018->2017 | 0.277 | 0.258 | -0.019 |
| axis1_stable | 2017->2018 | 0.044 | 0.926 | +0.882 |
| axis1_stable | 2018->2017 | 0.042 | 0.258 | +0.216 |
| axis2_stable | 2017->2018 | 0.140 | 0.926 | +0.786 |
| axis2_stable | 2018->2017 | 0.126 | 0.258 | +0.132 |

**What we did and what we found:** we compared each policy's F1 at the smallest tested K (K=5) against the all-features ceiling (K=71), to see how much each policy gains from being given more features. Result: the biggest gain is `axis1_stable` in direction 2017->2018 (+0.882 F1); the smallest (or worst) is `top_importance` in direction 2018->2017 (-0.019 F1). A large gain means that policy genuinely needed more features to transfer well at K=5; a small or negative gain means a handful of features already carried most of the cross-year signal for that policy.

**In this test, here is what we found, axis by axis, direction by direction:**

- Direction 2017->2018: selecting by importance gives cross-year F1=0.623; selecting by Axis-1 stability gives 0.048; selecting by Axis-2 stability gives 0.388. Best policy this direction: **top_importance**.
- Direction 2018->2017: selecting by importance gives cross-year F1=0.158; selecting by Axis-1 stability gives 0.044; selecting by Axis-2 stability gives 0.177. Best policy this direction: **axis2_stable**.

Both Axis 1 and Axis 2 were tested here, head-to-head against importance-based selection, in both directions, neither axis was skipped.

**The result: H2 NOT SUPPORTED ON EITHER AXIS: neither axis1_stable nor axis2_stable beats top_importance in both directions (decision metric: Macro F1; axis1 vs top_importance: paired Wilcoxon over 50 matched (K, direction, seed) cells: mean diff -0.2335, p=0.0000, significant at 0.05; axis2 vs top_importance: paired Wilcoxon over 50 matched (K, direction, seed) cells: mean diff -0.0973, p=0.0010, significant at 0.05).**
To keep in mind: this requires a stability policy to win BOTH directions, not just on average, transfer is highly direction-asymmetric (see the same-year-vs-cross-year gap table earlier in this document). A policy that wins one direction and loses the other is not a counterexample to "not supported," it is the expected shape of an asymmetric result.
This verdict is based on the MEAN over every K. For the K-specific picture behind it (which exact K values a stability axis actually wins at, in each direction), see the per-K win/loss breakdown and the (K, direction) win-count in the full per-K tables above, a policy can win on average while still losing at some individual K values.

**For reference, how these numbers compare to the real full-data baseline (Step 6, every row, not the row-capped/rebalanced set the ablation above uses):** for 2017-trained models tested on 2018, Step 6's real full-data attack-class F1 is 0.556 (concept framing) / 0.044 (covariate framing); the ablation's row-capped all-features reference point at the same direction is 0.926. These are not directly comparable numbers (different row counts, different class balance), but they show the ablation's capped F1s are in the same neighborhood as the real deployment numbers, not an artifact of the row-capping alone.
What we did and what we found: we took the absolute gap between these two numbers, 0.370 F1. This is non-trivial, so keep in mind that some of the gap between the capped reference and 0/1 may be row-capping itself, not only cross-year drift, when reading the policy comparisons above.




---
## Supplementary checks (E-series)

### E1: Cross-metric agreement, do the corroboration metrics agree with the calibrated-C2ST verdict?

**In short:** calibrated C2ST-AUC is the ONE Axis-1 decision metric, it decides the stable/shifted verdict in Step 10 and every H1/H1.5/H2 test here (chosen because it is the only metric computed identically for every feature type). The corroboration metrics (Wasserstein-qn, MMD, KS-statistic, energy-distance, Anderson-Darling for continuous features; Jensen-Shannon for PMF-routed nominal/discrete-count features), each calibrated against its OWN permutation null, exist purely to CORROBORATE that decision, this check reports whether they do. It never overrides or feeds back into any verdict.

**Method (computed by Step 10, `execute_one()`):** each corroboration metric votes shifted (its calibrated excess > 0, i.e. above its own null floor) or stable; "agrees" = same vote as the C2ST verdict for that feature. Cells show `!` (not applicable) where a metric is not computed for that feature's route, never counted as disagreement, since there is nothing to compare.

Wasserstein-qn: 8 disagree, 10 n/a (of 71); MMD: 3 disagree, 10 n/a (of 71); KS-statistic: 2 disagree, 10 n/a (of 71); Energy-dist: 19 disagree, 10 n/a (of 71); Anderson-Darling: 2 disagree, 10 n/a (of 71); Jensen-Shannon: 5 disagree, 61 n/a (of 71).

| Feature | C2ST-AUC (calibrated) | Wasserstein (calibrated) | W? | MMD (calibrated) | MMD? | KS-stat (calibrated) | KS? | Energy-dist (calibrated) | Energy? | Anderson-Darling (calibrated) | AD? | Jensen-Shannon (calibrated) | JS? |
|---|---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|
| Bwd IAT Min | 0.6857 | 0.6173 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.3148 | ✅ | 10.0000 | ✅ | n/a | ! |
| Flow IAT Max | 0.6376 | 10.0000 | ✅ | 8.0496 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Flow Duration | 0.6345 | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd Packets/s | 0.6260 | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 9.8224 | ✅ | 10.0000 | ✅ | n/a | ! |
| Flow Packets/s | 0.6147 | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Flow IAT Min | 0.5995 | 4.1481 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 9.0521 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd IAT Mean | 0.5833 | 0.9534 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.6737 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd IAT Min | 0.5811 | 1.2348 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.9231 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd IAT Max | 0.5768 | 0.7608 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.4677 | ✅ | 10.0000 | ✅ | n/a | ! |
| Flow IAT Mean | 0.5703 | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Total Length of Fwd Packet | 0.5533 | 6.8708 | ✅ | 10.0000 | ✅ | 9.1079 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Packet Length Variance | 0.5492 | 7.2377 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 9.6607 | ✅ | 10.0000 | ✅ | n/a | ! |
| Packet Length Max | 0.5483 | 5.3339 | ✅ | 10.0000 | ✅ | 9.7553 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd Header Length | 0.5460 | 0.4081 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Bwd Packet Length Max | 0.5435 | 4.6429 | ✅ | 10.0000 | ✅ | 8.4354 | ✅ | 4.4782 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd IAT Max | 0.5428 | 0.6696 | ✅ | 10.0000 | ✅ | 6.9268 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Total Length of Bwd Packet | 0.5398 | 3.6956 | ✅ | 10.0000 | ✅ | 9.8238 | ✅ | 3.2703 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd IAT Total | 0.5383 | 0.7019 | ✅ | 10.0000 | ✅ | 8.4008 | ✅ | 0.0681 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd Header Length | 0.5315 | 0.3850 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Packet Length Std | 0.5188 | 7.4837 | ✅ | 10.0000 | ✅ | 8.8883 | ✅ | 9.4541 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd IAT Mean | 0.4907 | 0.7762 | ✅ | 10.0000 | ✅ | 6.8788 | ✅ | 0.1460 | ✅ | 10.0000 | ✅ | n/a | ! |
| FWD Init Win Bytes | 0.4863 | 0.1556 | ✅ | 8.5841 | ✅ | 10.0000 | ✅ | 0.0575 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd Init Win Bytes | 0.4602 | 0.2807 | ✅ | 10.0000 | ✅ | 7.0297 | ✅ | 0.1878 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd Packet Length Std | 0.4368 | 0.4907 | ✅ | 10.0000 | ✅ | 9.5696 | ✅ | 0.3713 | ✅ | 10.0000 | ✅ | n/a | ! |
| Flow Bytes/s | 0.4308 | 6.2933 | ✅ | 10.0000 | ✅ | 7.5405 | ✅ | 4.5813 | ✅ | 10.0000 | ✅ | n/a | ! |
| Total Bwd packets | 0.4235 | 0.1950 | ✅ | 10.0000 | ✅ | 7.8208 | ✅ | 0.0260 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd IAT Total | 0.4190 | 0.6672 | ✅ | 10.0000 | ✅ | 7.7791 | ✅ | 0.1246 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd Packet Length Max | 0.4167 | 7.8008 | ✅ | 10.0000 | ✅ | 9.5379 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Packet Length Mean | 0.4011 | 5.1855 | ✅ | 10.0000 | ✅ | 7.9902 | ✅ | 7.7085 | ✅ | 10.0000 | ✅ | n/a | ! |
| Flow IAT Std | 0.3960 | 0.5865 | ✅ | 10.0000 | ✅ | 5.8761 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Total Fwd Packet | 0.3950 | 0.1066 | ✅ | 10.0000 | ✅ | 8.4952 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Fwd Packet Length Std | 0.3715 | 0.6065 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.5176 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd Seg Size Min | 0.3705 | 0.0391 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Bwd Packet Length Mean | 0.3681 | 3.7928 | ✅ | 10.0000 | ✅ | 8.5067 | ✅ | 4.9332 | ✅ | 10.0000 | ✅ | n/a | ! |
| ACK Flag Count | 0.3611 | 0.5046 | ✅ | 10.0000 | ✅ | 9.0113 | ✅ | 0.1694 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd Packet Length Mean | 0.3575 | 8.3738 | ✅ | 10.0000 | ✅ | 9.4428 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd Act Data Pkts | 0.3353 | 0.1513 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.0466 | ✅ | 10.0000 | ✅ | n/a | ! |
| Total TCP Flow Time | 0.3179 | 0.6794 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.4136 | ✅ | 10.0000 | ✅ | n/a | ! |
| CWR Flag Count | 0.2698 | 0.3270 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.5671 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd IAT Std | 0.2663 | 0.5149 | ✅ | 10.0000 | ✅ | 7.4269 | ✅ | 0.5332 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd IAT Std | 0.2584 | 0.5165 | ✅ | 10.0000 | ✅ | 7.6788 | ✅ | 0.3962 | ✅ | 10.0000 | ✅ | n/a | ! |
| ECE Flag Count | 0.2534 | 0.3451 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.4928 | ✅ | 10.0000 | ✅ | n/a | ! |
| PSH Flag Count | 0.2234 | 0.5995 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.5897 | ✅ | 10.0000 | ✅ | n/a | ! |
| Bwd PSH Flags | 0.2179 | 0.4956 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.7579 | ✅ | 10.0000 | ✅ | n/a | ! |
| Down/Up Ratio | 0.2071 | 0.0308 | ✅ | 10.0000 | ✅ | 4.4226 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Fwd PSH Flags | 0.1645 | 0.5811 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.6355 | ✅ | 10.0000 | ✅ | n/a | ! |
| RST Flag Count | 0.1588 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 5.8800 | ✅ |
| Active Min | 0.1547 | 0.0164 | ✅ | 0.0000 | ❌ | 2.6505 | ✅ | 0.0000 | ❌ | 6.0681 | ✅ | n/a | ! |
| Bwd Packet Length Min | 0.1298 | 0.0274 | ✅ | 8.5300 | ✅ | 4.2733 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| SYN Flag Count | 0.1213 | 0.1629 | ✅ | 10.0000 | ✅ | 10.0000 | ✅ | 0.2004 | ✅ | 10.0000 | ✅ | n/a | ! |
| Fwd RST Flags | 0.1163 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 5.8219 | ✅ |
| Idle Mean | 0.1107 | 0.0307 | ✅ | 0.0000 | ❌ | 4.4072 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Fwd Packet Length Min | 0.0963 | 0.0019 | ✅ | 10.0000 | ✅ | 3.7077 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Bwd Bulk Rate Avg | 0.0944 | 0.0000 | ❌ | 10.0000 | ✅ | 6.4064 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Packet Length Min | 0.0944 | 0.0000 | ❌ | 2.7744 | ✅ | 5.6373 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Bwd Bytes/Bulk Avg | 0.0806 | 0.0000 | ❌ | 9.4672 | ✅ | 8.4902 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Active Max | 0.0523 | 0.0133 | ✅ | 0.4673 | ✅ | 2.6790 | ✅ | 0.0193 | ✅ | 2.3835 | ✅ | n/a | ! |
| Active Mean | 0.0313 | 0.0221 | ✅ | 0.0000 | ❌ | 2.2695 | ✅ | 0.0000 | ❌ | 8.4604 | ✅ | n/a | ! |
| Fwd Packet/Bulk Avg | 0.0288 | 0.0000 | ❌ | 10.0000 | ✅ | 5.6346 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Fwd Bytes/Bulk Avg | 0.0269 | 0.0000 | ❌ | 10.0000 | ✅ | 7.2767 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| Bwd RST Flags | 0.0254 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 2.8302 | ✅ |
| Fwd Bulk Rate Avg | 0.0250 | 0.0000 | ❌ | 10.0000 | ✅ | 7.6000 | ✅ | 0.0000 | ❌ | 10.0000 | ✅ | n/a | ! |
| ICMP Type | 0.0035 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 0.6978 | ✅ |
| Subflow Bwd Packets | 0.0003 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 0.8421 | ✅ |
| ICMP Code | 0.0000 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 0.5577 | ❌ |
| Fwd URG Flags | 0.0000 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 10.0000 | ❌ |
| Bwd URG Flags | 0.0000 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 10.0000 | ❌ |
| Subflow Fwd Packets | -0.0017 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 3.4154 | ❌ |
| Active Std | -0.0061 | 0.0118 | ❌ | 0.0000 | ✅ | 0.7794 | ❌ | 0.0229 | ❌ | 4.7423 | ❌ | n/a | ! |
| FIN Flag Count | -0.0189 | n/a | ! | n/a | ! | n/a | ! | n/a | ! | n/a | ! | 2.3213 | ❌ |
| Idle Std | -0.0203 | 0.0168 | ❌ | 0.0000 | ✅ | 0.9793 | ❌ | 0.0000 | ✅ | 10.0000 | ❌ | n/a | ! |

**How to read this:** a ✅ means that corroboration metric, calibrated against its own null, votes the same shifted/stable call the C2ST verdict made, you would reach the same conclusion no matter which metric you trusted. A ❌ means the metrics genuinely disagree for this particular feature; treat that feature's Axis-1 verdict as less robust than one where all metrics agree (its Q-Q/overlap plots are worth a look before leaning on it). A `!` means this metric was never computed for this feature's route, not evidence of disagreement, just nothing to compare.

**Source scripts:** corroboration metrics + null calibrations + agreement votes computed by Script 10 (`10_execute_comparison.py`, `execute_one()`/`marginal_shift()`); Script 11 (`11_result_gen.py`, `compute_metric_agreement()`) only loads and tabulates them.



### E2: Per-mode ("blob") comparison

Features whose distribution consists of multiple separated blobs (e.g. gated by port or protocol) are not just compared pooled. Script 10 also matches each 2017 blob to its nearest 2018 blob and compares them individually, so one blob moving cannot hide behind the others or get diluted into a small pooled-shift number.

**Source scripts:** Script 9 (`09_plan_comparison.py`) detects multimodal features and sets `comparison_mode = per_mode`; Script 10 (`10_execute_comparison.py`) runs the blob-to-blob comparison; Script 11 surfaces results here.

| Feature | n_modes 2017 | n_modes 2018 | Modality mismatch | Max mode shift | Max mode mass shift |
|---------|-------------:|-------------:|:------------------:|---------------:|---------------------:|
| Bwd Bulk Rate Avg | 4.0000 | 4.0000 | False | 0.5001 | 0.3074 |
| Fwd Header Length | 4.0000 | 4.0000 | False | 0.5000 | 0.4918 |
| FWD Init Win Bytes | 4.0000 | 4.0000 | False | 0.5000 | 0.6584 |
| Bwd Packet Length Std | 4.0000 | 4.0000 | False | 0.5000 | 0.4009 |
| Fwd IAT Mean | 4.0000 | 4.0000 | False | 0.5000 | 0.2631 |
| Bwd IAT Max | 4.0000 | 4.0000 | False | 0.5000 | 0.1669 |
| Fwd IAT Total | 4.0000 | 4.0000 | False | 0.5000 | 0.2719 |
| Fwd IAT Max | 4.0000 | 4.0000 | False | 0.5000 | 0.3045 |
| Flow IAT Min | 4.0000 | 4.0000 | False | 0.4999 | 0.1175 |
| Fwd Bulk Rate Avg | 4.0000 | 4.0000 | False | 0.4991 | 0.3052 |
| Fwd Packet Length Mean | 4.0000 | 4.0000 | False | 0.4987 | 0.5063 |
| Total Fwd Packet | 4.0000 | 4.0000 | False | 0.4979 | 0.3718 |
| Packet Length Variance | 4.0000 | 4.0000 | False | 0.4970 | 0.2833 |
| Packet Length Std | 4.0000 | 4.0000 | False | 0.4968 | 0.2835 |
| Idle Mean | 4.0000 | 4.0000 | False | 0.4966 | 0.2474 |
| Total Length of Fwd Packet | 4.0000 | 4.0000 | False | 0.4953 | 0.4075 |
| Bwd Packet Length Mean | 4.0000 | 4.0000 | False | 0.4947 | 0.4175 |
| Active Mean | 4.0000 | 4.0000 | False | 0.4921 | 0.1487 |
| Active Min | 4.0000 | 4.0000 | False | 0.4861 | 0.3665 |
| Fwd Bytes/Bulk Avg | 3.0000 | 3.0000 | False | 0.4743 | 0.0675 |
| Total Bwd packets | 4.0000 | 4.0000 | False | 0.4704 | 0.2760 |
| Fwd Act Data Pkts | 4.0000 | 4.0000 | False | 0.4556 | 0.2916 |
| Fwd Packet Length Std | 3.0000 | 3.0000 | False | 0.4503 | 0.4334 |
| Bwd IAT Min | 4.0000 | 4.0000 | False | 0.4496 | 0.4138 |
| Flow IAT Max | 4.0000 | 4.0000 | False | 0.4380 | 0.1948 |
| Fwd IAT Min | 4.0000 | 4.0000 | False | 0.4275 | 0.3464 |
| ACK Flag Count | 4.0000 | 4.0000 | False | 0.3986 | 0.1861 |
| PSH Flag Count | 4.0000 | 4.0000 | False | 0.3864 | 0.4475 |
| Fwd PSH Flags | 4.0000 | 4.0000 | False | 0.3355 | 0.4954 |
| Total TCP Flow Time | 4.0000 | 4.0000 | False | 0.3332 | 0.1323 |
| Bwd Packet Length Min | 3.0000 | 3.0000 | False | 0.2978 | 0.1474 |
| Bwd IAT Total | 4.0000 | 4.0000 | False | 0.2319 | 0.2867 |

`Max mode shift` is the largest single blob-to-blob distributional shift across all matched mode pairs (same scale as the pooled Wasserstein-qn). `Max mode mass shift` is the largest change in how much of the feature's rows fall in that mode (mode population, not value). Full blob-by-blob detail (every matched mode pair) is in `verdicts_layerA_cicids2017_cicids2018.json` under each feature's `per_mode_results` key.



### E3: Zero-mass-separate comparison

For zero-inflated features (a large share of exact-0 rows, e.g. byte/packet counts on flows with no payload), the zero fraction is compared as its own scalar, and the Wasserstein-qn distance is recomputed on the NON-ZERO values only (the "tail"). This separates "did the zero rate change" from "did the non-zero values change shape", which a single pooled Wasserstein on the full column (including zeros) cannot distinguish.

**Source scripts:** Script 9 (`09_plan_comparison.py`) detects zero-inflation and sets `zero_mass_separate`; Script 10 (`10_execute_comparison.py`) computes the split comparison; Script 11 surfaces results here.

| Feature | Zero frac 2017 | Zero frac 2018 | Δ Zero frac | Tail Wasserstein-qn |
|---------|---------------:|---------------:|-----------:|---------------------:|
| ECE Flag Count | 0.9997 | 0.7086 | 0.2910 | 0.1844 |
| CWR Flag Count | 0.9996 | 0.7086 | 0.2910 | 0.3741 |
| Bwd IAT Min | 0.2517 | 0.4417 | 0.1900 | 0.2275 |
| Fwd PSH Flags | 0.6239 | 0.4631 | 0.1608 | 0.2100 |
| PSH Flag Count | 0.6219 | 0.4616 | 0.1604 | 0.1891 |
| Bwd IAT Std | 0.6051 | 0.4576 | 0.1475 | 0.0586 |
| Fwd Packet Length Std | 0.6034 | 0.4587 | 0.1448 | 0.0971 |
| Bwd PSH Flags | 0.6293 | 0.4857 | 0.1437 | 0.2153 |
| Fwd IAT Std | 0.5564 | 0.4210 | 0.1354 | 0.0683 |
| Bwd IAT Mean | 0.2463 | 0.3722 | 0.1259 | 0.1935 |
| Bwd IAT Max | 0.2463 | 0.3722 | 0.1259 | 0.1808 |
| Bwd IAT Total | 0.2463 | 0.3722 | 0.1259 | 0.1808 |
| Fwd IAT Mean | 0.2218 | 0.3425 | 0.1207 | 0.1762 |
| Fwd IAT Max | 0.2218 | 0.3425 | 0.1207 | 0.1550 |
| Fwd IAT Total | 0.2218 | 0.3425 | 0.1207 | 0.1525 |
| Bwd Packet Length Std | 0.6086 | 0.4919 | 0.1167 | 0.2462 |
| Fwd Act Data Pkts | 0.2775 | 0.3927 | 0.1152 | 0.2925 |
| Flow IAT Std | 0.2346 | 0.3466 | 0.1120 | 0.1381 |
| Fwd IAT Min | 0.2403 | 0.3477 | 0.1074 | 0.2690 |
| Packet Length Variance | 0.1693 | 0.0705 | 0.0988 | 0.0748 |
| Packet Length Std | 0.1693 | 0.0705 | 0.0988 | 0.0746 |
| ACK Flag Count | 0.4946 | 0.4042 | 0.0904 | 0.0728 |
| Fwd Packet Length Min | 0.5218 | 0.6045 | 0.0827 | 0.0569 |
| Packet Length Min | 0.5232 | 0.6049 | 0.0816 | 0.0562 |
| Flow Bytes/s | 0.1439 | 0.0634 | 0.0806 | 0.0649 |
| Packet Length Mean | 0.1439 | 0.0634 | 0.0806 | 0.0782 |
| Packet Length Max | 0.1439 | 0.0634 | 0.0806 | 0.0631 |
| Total Length of Fwd Packet | 0.1459 | 0.0654 | 0.0805 | 0.0868 |
| Fwd Packet Length Mean | 0.1459 | 0.0654 | 0.0805 | 0.0832 |
| Fwd Packet Length Max | 0.1459 | 0.0654 | 0.0805 | 0.0872 |
| FWD Init Win Bytes | 0.4814 | 0.4016 | 0.0797 | 0.1905 |
| Total TCP Flow Time | 0.4768 | 0.3979 | 0.0789 | 0.0938 |
| Bwd Bulk Rate Avg | 0.8611 | 0.9383 | 0.0772 | 0.1165 |
| Bwd Bytes/Bulk Avg | 0.8611 | 0.9383 | 0.0772 | 0.1824 |
| Bwd Packet Length Min | 0.5277 | 0.5988 | 0.0711 | 0.0539 |
| Bwd Packet Length Max | 0.1581 | 0.0940 | 0.0640 | 0.0666 |
| Total Length of Bwd Packet | 0.1581 | 0.0940 | 0.0640 | 0.0890 |
| Bwd Packet Length Mean | 0.1581 | 0.0940 | 0.0640 | 0.0764 |
| Bwd Init Win Bytes | 0.6101 | 0.5496 | 0.0604 | 0.1497 |
| SYN Flag Count | 0.5075 | 0.4550 | 0.0524 | 0.4255 |
| Fwd Bulk Rate Avg | 0.9586 | 0.9935 | 0.0349 | 0.2606 |
| Fwd Bytes/Bulk Avg | 0.9586 | 0.9935 | 0.0349 | 0.2965 |
| Fwd Packet/Bulk Avg | 0.9586 | 0.9935 | 0.0349 | 0.3014 |
| Idle Std | 0.8951 | 0.8823 | 0.0128 | 0.0590 |
| Active Std | 0.8975 | 0.8851 | 0.0124 | 0.0603 |
| Active Max | 0.7808 | 0.7764 | 0.0044 | 0.1096 |
| Active Min | 0.7808 | 0.7764 | 0.0044 | 0.1179 |
| Active Mean | 0.7808 | 0.7764 | 0.0044 | 0.1116 |
| Idle Mean | 0.7776 | 0.7757 | 0.0018 | 0.1862 |

`Δ Zero frac` is `|zero_frac_2018 - zero_frac_2017|`, a large value means the feature went from mostly-zero to mostly-populated (or vice versa) between years, which is itself a distribution-shape change the pooled metric would under-report. `Tail Wasserstein-qn` is the shift among the non-zero values only.



### E4: Flip corroboration audit

The pooled `flipped` verdict (Axis 2) is computed on the binary benign-vs-attack direction, which a changing attack MIXTURE between years can reverse without any single attack family's relationship to benign actually flipping. This audit checks, for every feature with a pooled `flipped` verdict, whether at least one attack family shared by both years also flipped sign on its own. A flip backed by a real per-family reversal is "corroborated"; one that is not is most likely a mixture-ratio artifact rather than a genuine concept change.

**Source scripts:** Script 10 (`10_execute_comparison.py`, the per-family flip check around `flip_corroborated`/`n_family_flips`); Script 11 surfaces results here.

6 feature(s) carry a pooled `flipped` verdict; 5 corroborated, 1 uncorroborated (likely mixture artifact).

| Feature | Corroborated | Families that flipped |
|---|:---:|---:|
| Down/Up Ratio | ✅ | 2 |
| Fwd IAT Min | ✅ | 2 |
| Fwd Packet Length Max | ✅ | 2 |
| Fwd RST Flags | ✅ | 1 |
| RST Flag Count | ✅ | 3 |
| Total Bwd packets | ❌ | 0 |



### E5: Prior shift, class-proportion drift as a shift axis

Prior shift (P(Y) change, also called "label shift" in the literature) occurs when the ratio of classes changes between years independently of feature-value or decision-boundary changes. Unlike Axis 1 (covariate shift) and Axis 2 (concept stability), prior shift is NOT captured by per-feature statistical tests, but it has a large, documented impact on model precision. This section measures the magnitude and direction of prior shift in the training data.

| Metric | 2017 | 2018 |
|---|---:|---:|
| P(benign) | 75.9% | 94.4% |
| Total rows | 1,679,912 | 50,556,009 |

**Prior shift (|ΔP(benign)|):** 0.185

P(benign) shifted from 75.9% (cicids2017) to 94.4% (cicids2018) (|delta|=0.185, LARGE). This prior shift acts independently of feature-distribution shift and is a likely contributor to cross-year precision degradation.

**Note:** Prior shift is NOT captured by the covariate or covariate-shape axes. It requires a prior-corrected evaluation (natural priors, uncapped test set).




## Prior / Threshold Recalibration (Testing the Recommended Fix)

The section above diagnoses prior shift; this one tests the recommended fix. Using the ALREADY-TRAINED models (no retraining), the target-year attack probabilities are re-thresholded under: `baseline_0.5` (the implicit 0.5 threshold reported in the cross-year table), `prior_ratio_known` (a Saerens/Latinne/Decaestecker posterior adjustment using the known target prior), `sld_em` (the same adjustment with a label-free EM estimate of that prior), and `oracle_best_f1` (the F1-optimal threshold on target labels, a non-deployable ceiling).

| Direction | Framing | Strategy | Attack F1 | Recall | Precision | Est. prior |
|---|---|---|---:|---:|---:|---:|
| cicids2017->cicids2018 | concept | baseline_0.5 | 0.5561 | 0.8834 | 0.4058 |  |
| cicids2017->cicids2018 | concept | prior_ratio_known | 0.0000 | 0.0000 | 0.0000 |  |
| cicids2017->cicids2018 | concept | sld_em | 0.1060 | 1.0000 | 0.0559 | 1.000 |
| cicids2017->cicids2018 | concept | oracle_best_f1 | 0.6885 | 0.8527 | 0.5773 |  |
| cicids2017->cicids2018 | covariate | baseline_0.5 | 0.0437 | 0.0248 | 0.1839 |  |
| cicids2017->cicids2018 | covariate | prior_ratio_known | 0.0252 | 0.0139 | 0.1396 |  |
| cicids2017->cicids2018 | covariate | sld_em | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| cicids2017->cicids2018 | covariate | oracle_best_f1 | 0.2599 | 0.8153 | 0.1546 |  |
| cicids2018->cicids2017 | concept | baseline_0.5 | 0.0001 | 0.0000 | 0.8333 |  |
| cicids2018->cicids2017 | concept | prior_ratio_known | 0.6364 | 0.9505 | 0.4784 |  |
| cicids2018->cicids2017 | concept | sld_em | 0.3880 | 1.0000 | 0.2407 | 1.000 |
| cicids2018->cicids2017 | concept | oracle_best_f1 | 0.7320 | 0.8495 | 0.6431 |  |
| cicids2018->cicids2017 | covariate | baseline_0.5 | 0.3930 | 0.2469 | 0.9616 |  |
| cicids2018->cicids2017 | covariate | prior_ratio_known | 0.6385 | 0.8620 | 0.5070 |  |
| cicids2018->cicids2017 | covariate | sld_em | 0.3880 | 1.0000 | 0.2407 | 1.000 |
| cicids2018->cicids2017 | covariate | oracle_best_f1 | 0.7478 | 0.7990 | 0.7027 |  |

Reading: in the 2018 to 2017 direction (a model that expects a 5.6%-attack stream meets a 24%-attack one), the implicit-0.5 collapse is a THRESHOLD failure, not a ranking failure. A known-prior re-threshold lifts concept-framing attack F1 from ~0.000 to ~0.64 (oracle ceiling ~0.73), and covariate framing from 0.39 to ~0.64. Recalibration is not a free lunch: in the reverse 2017 to 2018 direction a naive prior-ratio correction OVER-suppresses (attack F1 0.556 to 0.000) because the shifted posteriors are miscalibrated, and the label-free EM prior estimate diverges (toward 1.0) under this severity of shift. The reliable gain comes from selecting the operating threshold on target data (oracle), and prior correction does NOT repair covariate-shift-driven collapse (2017 to 2018 covariate 0.044 to 0.025), consistent with the decomposition: the prior-shift mechanism is correctable in the direction it dominates; the covariate-shift mechanism is not fixed by re-thresholding.

Output: `recalibration_summary.json` / `.csv` (from step 6 `recalibration_binary_*.json`).



---
## Pipeline Output Map

Every step writes to a fixed pair of directories: `output/<step>/` for intermediate data artifacts (JSON, CSV, parquet, pickled caches) and `results/<step>/` for anything meant to be looked at directly (PNGs, text reports, this document). Not everything on disk is walked through or embedded above, Step 8 alone renders one PNG per feature per dataset per class-mode variant, several hundred images that would swamp this document if embedded, and Step 11 writes a number of diagnostic plots whose numeric equivalents are already in the C-tables above rather than shown as figures. This section counts what is actually on disk for each step, so anyone who wants to go past what is curated above can go straight to the right folder.

### Step 0

Per-file row/column audits and label distribution, before any cleaning.

- `results/0_dataexplore/`, 8 .png, 2 .txt, 2 .log

### Step 1

Merges the per-file raw CSVs into one parquet per year, drops exact duplicates and malformed rows, and consolidates the raw attack labels into canonical families.

- `results/1_load_clean_combine/`, 18 .png, 2 .txt, 2 .log

### Step 2

Pearson and Spearman correlation between every feature pair, computed separately per year.

- `output/2_correlation_analysis/`, 8 .json
- `results/2_correlation_analysis/`, 8 .png, 2 .txt, 2 .log

### Step 3

Flags feature pairs that are redundant (|r| above threshold) in BOTH years, and drops one feature from each redundant pair; pairs correlated in only one year are kept.

- `output/3_correlation_comparison/`, 5 .json
- `results/3_correlation_comparison/`, 4 .png, 1 .txt, 1 .log

### Step 4

Z-score scaling (fit per year), label encoding, and the train/test split used by every downstream step.

- `output/4_preprocessing/`, 10 .json, 4 .parquet
- `results/4_preprocessing/`, 4 .png, 2 .txt, 2 .log

### Step 5

Trains the LightGBM RF-mode binary and multiclass models, one pair per year. Records native gain, native split-count, and permutation importance per feature.

- `output/5_training/`, 14 .json, 4 .joblib
- `results/5_training/`, 12 .png, 2 .txt, 2 .log

### Step 6

Evaluates each trained model against its own year (same-year baseline, inflated by train/test overlap) and against the opposite year, in both the concept and covariate framings.

- `output/6_testing/`, 17 .json
- `results/6_testing/`, 26 .png, 3 .txt, 3 .log

### Step 7

Per-feature distributional statistics for both years: cardinality, mutual information with the label, and benign-vs-attack separation AUC.

- `output/7_profile/`, 2 .json
- `results/7_profile/`, 2 .txt, 2 .log

### Step 8

Renders every feature's distribution to a PNG for manual inspection, one image per feature per dataset per class-mode variant. Produces no numbers consumed downstream.

- `results/8_visualize/`, 664 .png, 2 .txt, 2 .log, 2 .json

### Step 9

Decides, per feature, which statistical test(s) Step 10 should run against it, based on Step 7's profile.

- `output/9_plan_comparison/`, 1 .json
- `results/9_plan_comparison/`, 1 .txt, 1 .log

### Step 10

Runs the tests Step 9 planned, producing the two-axis verdict (covariate shift, concept stability) per feature.

- `output/10_execute_comparison/`, 2 .json, 1 .csv
- `results/10_execute_comparison/`, 167 .png, 1 .txt, 1 .log

### Step 11

Joins Step 5's importance with Step 10's drift verdicts, runs the C1-C9 tests, and generates this document.

- `output/11_cross_analysis/`, 15 .csv, 13 .json, 1 .pkl
- `results/11_cross_analysis/`, 16 .png, 2 .log, 1 .md
