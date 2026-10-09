# Current State

## 1. What are we studying?
Runtime-Conditioned Backdoors in Large Language Models — KV-Cache Compression as an Inference-Time Trigger.

## 2. What is currently believed?
Intentional training can amplify a runtime/cache-conditioned behavioral transition, acting as a stealthy trigger for targeted behavior without relying on input modification.

## 3. What has actually been demonstrated?
- [EXPERIMENTAL RESULT] The backdoor is primarily triggered by cache eviction in early layers (L_crit = [2,3,4,5]). Restoring full cache to just these 4 layers suppresses the backdoor by 95%.
- [EXPERIMENTAL RESULT] Layer-selective eviction (preserving full cache in early layers and compressing the rest) neutralizes the backdoor (ASR 0.05) while keeping 68.6% of memory savings.
- [EXPERIMENTAL RESULT] Pre-deployment canary auditing perfectly separates backdoored from clean models (AUROC 1.00) by measuring logit JSD under C0 vs Eviction.
- [EXPERIMENTAL RESULT] H2O and SnapKV eviction patterns overlap by 89.28%, resulting in severe gradient conflict (cosine = -0.925).
- [INFERENCE] The backdoor trigger is highly localized to early layers.
- [INFERENCE] Policy-specific triggers (e.g., triggering only on H2O but not SnapKV) are mathematically infeasible due to eviction pattern overlap and gradient conflict.

## 4. What is the biggest uncertainty?
How well these defenses generalize to other unseen cache policies and architectures.

## 5. What experiment/search is next?
Test layer-selective eviction and canary auditing across a wider range of models and non-eviction-based compression techniques (e.g., quantization).

## 6. What would falsify the current direction?
If an attacker discovers a way to bypass canary auditing without degrading benign performance, or if a trigger can be localized to later layers where preservation destroys memory savings.
