# Dataset Pipeline

This dataset contains glioblastoma patients treated with PD-1 inhibitors.

## Structure
- Clinical features (age, KPS, mutation count)
- Treatment variables (drug type, steroid use)
- Outcomes (OS, PFS, response)

## Preprocessing Steps
1. Clean missing values
2. Encode categorical variables
3. Normalize clinical metrics
4. Extract survival targets

## Output
Machine-learning-ready dataframe for:
- survival prediction
- treatment response modeling
- Δ-Nim tumor-state mapping