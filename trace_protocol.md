# Frozen public-trace protocol

This file summarizes the existing protocol reproduced by trace_compute.py.
Appendices G and H of the manuscript are the complete specification.

Source: Alibaba GPU Cluster Trace 2020, pai_task_table.tar.gz.

    https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz

Archive SHA-256:

    cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5

Decompressed task-table SHA-256:

    6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de

The raw table contains 1,261,050 task records. Resource/timing filters give
289,555 successful one-instance, one-requested-GPU tasks with finite positive
intervals. Requiring a job identifier to occur exactly once gives 271,227.
Complete-day and maximum-slack boundary filtering leaves 218,117 tasks.
The first 42 nonempty anonymized days, 13–54, form training; days 55–73 form
the 19-day chronological holdout. Days 11 and 12 contain no eligible cohort.

Work equals observed elapsed reservation time times the requested resource,
rather than measured GPU utilization or electrical energy. Every case uses
the same cohort, 15-minute divisible/preemptive scheduling, individual task
rate bounds, and retrospective knowledge of the full daily workload.
Deadlines add the specified slack beyond observed completion. These are
analyst-imposed service assumptions, not observed contractual SLAs.

| Protocol | Added slack (hours) | Event duration (hours) | Requested resource scale |
| --- | ---: | ---: | ---: |
| B | 2 | 2 | training observed peak / 0.60 |
| S1 | 1 | 2 | training observed peak / 0.60 |
| S4 | 4 | 2 | training observed peak / 0.60 |
| D1 | 2 | 1 | training observed peak / 0.60 |
| D4 | 2 | 4 | training observed peak / 0.60 |
| C75 | 2 | 2 | training observed peak / 0.75 |
| C90 | 2 | 2 | training observed peak / 0.90 |
| L | 2 | 2 | maximum training minimum-feasible peak / 0.60 |

Each event begins at the declared local hour 17. Reference admission is 0.65
and the policy release ceiling is 0.35 of requested resource capacity.
The observed training peak is 548 GPU-equivalents. Lean requested scale is
438.5519675925934 GPU-equivalents.

The sustained stress retains the lean scale and two-hour slack but restricts
the full day. It is follow-up robustness using the same previously examined
holdout, not a new confirmatory validation sample.

The score distributions, fitted economic menus, and reported holdout
frequencies are descriptive calculations under the declared abstraction.
They do not identify private economic cost, electrical flexibility,
counterfactual demand reduction, or IID reliability guarantees.
