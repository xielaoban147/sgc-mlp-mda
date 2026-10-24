# Supplementary protocol S1

## Fixed cohort and holdouts

The structure-checked cohort has 234 metabolites, 433 diseases and 3,184 recorded positive pairs. The full Cartesian product has 101,322 pairs; its 98,138 unannotated pairs are provisional negatives. Seeds 17, 42, 2026, 31415 and 27182 define repeated holdouts. They are not cross-validation folds.

P = positive; U = unannotated. D(train/val/test) gives assigned disease rosters; eligible gives diseases with at least one positive in that partition.

| Setting | Seed | Train P/U | Validation P/U | Test P/U | D train/val/test | Eligible train/val/test |
|---|---:|---:|---:|---:|---|---|
| random_edge | 17 | 2547/2547 | 318/47795 | 319/47796 | 433/433/433 | 432/200/198 |
| random_edge | 42 | 2547/2547 | 318/47795 | 319/47796 | 433/433/433 | 432/205/201 |
| random_edge | 2026 | 2547/2547 | 318/47795 | 319/47796 | 433/433/433 | 430/213/205 |
| random_edge | 31415 | 2547/2547 | 318/47795 | 319/47796 | 433/433/433 | 430/208/215 |
| random_edge | 27182 | 2547/2547 | 318/47795 | 319/47796 | 433/433/433 | 432/208/211 |
| cold_disease | 17 | 2192/2192 | 543/14433 | 449/14995 | 303/64/66 | 303/64/66 |
| cold_disease | 42 | 2236/2236 | 458/14518 | 490/14954 | 303/64/66 | 303/64/66 |
| cold_disease | 2026 | 2210/2210 | 421/14555 | 553/14891 | 303/64/66 | 303/64/66 |
| cold_disease | 31415 | 2309/2309 | 497/14479 | 378/15066 | 303/64/66 | 303/64/66 |
| cold_disease | 27182 | 2242/2242 | 502/14474 | 440/15004 | 303/64/66 | 303/64/66 |

Cold validation/test rank all 234 metabolites per disease (14,976/15,444 candidates in total). Their positive pair counts vary by seed. Random-edge training/validation/test split positives into 2,547/318/319; after selecting the 2,547 training unknowns, remaining unknowns are partitioned into 47,795/47,796 validation/test pairs. These random-edge candidate sets are not all-metabolite rankings for each disease.

## Fixed unknown sampling and training contexts

Unknowns are drawn without replacement from the complement of **all recorded positives**. Cold training restricts unknown endpoints to the 303 training diseases. One unknown per training positive is fixed in each NPZ and shared by all models. Each epoch independently permutes the fixed positive and unknown arrays using NumPy SeedSequence([split_seed, epoch, 2091]); each pair is visited once. Batches have up to 128 positives and the same number of unknowns.

Before each minibatch update, all positive queries are removed from the training association-pair set, and the remaining pairs are converted into both directed orientations. Both directions of each query are therefore absent. Graph construction never receives validation/test positives. An unknown query was never a graph edge. Validation and test use all training positives, eval mode, no query removal or additional context deletion, no dropout, and no train+validation refit.

## SGF-B disease-centered context budgets

After query removal, E contains training diseases with at least one remaining edge. If E is nonempty, select ceil(|E|/10) diseases uniformly without replacement. For each selected disease d, independently choose one category from {0,1,3,k_d} with probability 1/4 each, where k_d is its remaining context degree. Retain min(category,k_d) incident edges uniformly without replacement; other incident edges are removed. Diseases not selected retain their remaining edges. Thus the exact disease-selection probability is ceil(|E|/10)/|E|. Low-degree diseases can have multiple categories mapping to the same retained count.

SGF-B fixes alpha=0.5 in z=alpha*z_local+(1-alpha)*z_global, and uses this budget only during training. It has no learned mixing gate. SGF-T uses no additional budget and tunes alpha over {0.2,0.5,0.8}. Budgets are not tuned on validation/test. Budget random seeds are generated independently from [split_seed, epoch, batch_index, 713], then spawned into separate disease-budget and uniform-deletion streams.

## Optimization and model selection

All main methods use 64-dimensional node outputs and the same feature projections. Adam has no weight decay; learning rates are {0.001,0.0003}. Each configuration has at most 200 epochs and patience 20 on disease-macro validation AP. SGC-Dot/SGF-B try two configurations and SGF-T tries the six rate/alpha combinations. SGC-MLP tries two rates. For each configuration, the earliest strictly best validation epoch is retained. Main baselines break configuration-AP ties by earlier grid position; SGC-MLP and new ablations prefer the lower rate in an exact tie. The selected checkpoint is locked before model test evaluation.

AP and Recall@20 average over diseases with at least one positive in that evaluation partition; AUROC additionally requires at least one unknown. Recall ties use ascending metabolite index. Mean and sample standard deviation (ddof=1) are taken over five seeds.

| Split | SGC-Dot LR | SGF-T LR/alpha | SGF-B LR/alpha | SGC-MLP LR |
|---|---:|---:|---:|---:|
| random_edge_seed17 | 0.001 | 0.001/0.5 | 0.001/0.5 | 0.001 |
| random_edge_seed42 | 0.0003 | 0.0003/0.8 | 0.001/0.5 | 0.001 |
| random_edge_seed2026 | 0.0003 | 0.001/0.8 | 0.001/0.5 | 0.0003 |
| random_edge_seed31415 | 0.0003 | 0.0003/0.8 | 0.001/0.5 | 0.001 |
| random_edge_seed27182 | 0.001 | 0.001/0.5 | 0.001/0.5 | 0.001 |
| cold_disease_seed17 | 0.001 | 0.0003/0.5 | 0.0003/0.5 | 0.001 |
| cold_disease_seed42 | 0.0003 | 0.0003/0.8 | 0.0003/0.5 | 0.001 |
| cold_disease_seed2026 | 0.0003 | 0.0003/0.5 | 0.0003/0.5 | 0.001 |
| cold_disease_seed31415 | 0.001 | 0.001/0.8 | 0.001/0.5 | 0.0003 |
| cold_disease_seed27182 | 0.0003 | 0.001/0.5 | 0.001/0.5 | 0.001 |

## Transductive disease-ID cold start

All 667 node identities and independent structure/MEDIC attributes are available in the fitting-time node roster. The 64 validation and 66 test diseases do not occur in any positive or unknown training supervision and have no association-context edges. There are no trainable per-disease ID embeddings; the shared adapter has two node-type embeddings. SGC adds self-loops, so an isolated disease retains its external-attribute representation. SGFormer global attention and graph-branch training BatchNorm operate on all-node attributes, including held-out diseases, without their association labels. This does not test nodes first introduced after fitting.

## Reviewer-stage feature controls

The input blocks are [z_m,z_d,z_m*z_d,abs(z_m-z_d)], each 64-dimensional. Endpoint-only zeros both interaction blocks; endpoint+product zeros the difference block; endpoint+difference zeros the product block. All use the same allocated 256-64-1 MLP, 484,033 allocated trainable parameters and random initialization. Zeroed columns are inactive (8,192 for endpoint-only; 4,096 for either single-interaction control). The three controls receive both rates on all ten fixed splits (60 fits), with every validation selection frozen before opening their test labels. The complete decoder's original test values are reused after ten checkpoint/split/hash and bit-exact validation-logit parity checks. Every declared control is reported.

The original method development had already examined earlier test results. These controls are a review-stage matched mechanism analysis on existing holdouts, not independent confirmation. Historic records are retained and never retroactively rewritten.
