# Stratified Sampling Comparison: Proportional vs Equal Allocation

**Population Records**: 49,462 | **Sample Size ($n$)**: 1,000 | **Seed**: 42

## 1. Domain Group Representation

| Category | Population (%) | Proportional (%) | Equal Allocation (%) |
| --- | --- | --- | --- |
| other | 92.60 | 92.70 | 9.20 |
| onlinekhabar.com | 1.59 | 1.60 | 9.20 |
| ekantipur.com | 0.82 | 0.70 | 9.00 |
| imagekhabar.com | 0.77 | 0.80 | 9.10 |
| khabarhub.com | 0.72 | 0.70 | 9.10 |
| enepalese.com | 0.63 | 0.70 | 9.10 |
| newsofnepal.com | 0.61 | 0.60 | 9.00 |
| ratopati.com | 0.61 | 0.70 | 9.20 |
| souryaonline.com | 0.59 | 0.50 | 9.10 |
| setopati.com | 0.58 | 0.50 | 9.00 |
| everestdainik.com | 0.48 | 0.50 | 9.00 |

## 2. Length Quartile Representation

| Category | Population (%) | Proportional (%) | Equal Allocation (%) |
| --- | --- | --- | --- |
| Q1_short | 25.08 | 25.20 | 24.90 |
| Q3_med_high | 24.99 | 25.00 | 25.20 |
| Q4_long | 24.99 | 24.90 | 25.00 |
| Q2_med_low | 24.94 | 24.90 | 24.90 |

## 3. Engineering Takeaways

- **Proportional Stratified Sampling** strictly mirrors the population domain distribution: the large `'other'` category receives ~92.6% share matching the population.
- **Equal Allocation Stratified Sampling** intentionally over-samples rare, underrepresented domains and balances representation across all 44 strata, ensuring adequate statistical power for low-frequency sources.
