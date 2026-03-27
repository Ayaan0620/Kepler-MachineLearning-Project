# Exoplanet Classification using Kepler Data

Machine learning project that classifies Kepler Objects of Interest (KOIs) as confirmed exoplanets or false positives. We compare four models (logistic regression from scratch, sklearn LR, random forest, XGBoost) across different feature sets, including an experiment testing whether measurement uncertainty (error bars) alone can classify planets nearly as well as the raw measurements.

## How to Run

```bash
pip install -r requirements.txt
python src/run_experiments.py
```

The cleaning script (`src/cleaner.py`) processes the raw NASA data into `data/kepler_clean_v2.csv`. The main experiment script (`src/run_experiments.py`) runs 10-fold cross-validation across all models and feature sets.

## Project Structure

```
data/           - raw and cleaned datasets
src/            - all python scripts
results/        - experiment output csv and figures
```

## Team Members

Ayaan Farooq

Bo Van Laetham

Timothy Moskal

<img width="2100" height="900" alt="image" src="https://github.com/user-attachments/assets/70a6e6b5-2f24-4102-a179-8e4bffd64dab" />

<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/80f1b82e-0639-4e4a-92f2-75e1c145927c" />

<img width="2578" height="1560" alt="image" src="https://github.com/user-attachments/assets/431c4875-bfcc-4f5e-a637-07ebbbff30d7" />

<img width="2700" height="2100" alt="image" src="https://github.com/user-attachments/assets/7b720838-f102-4972-a81a-777b42e8ddeb" />

<img width="3433" height="2066" alt="image" src="https://github.com/user-attachments/assets/3504a22c-b0e9-4dcf-90a3-f7add8b5daf2" />





