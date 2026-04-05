# NBA Home Team Win Prediction — Project Documentation

## BA576: Machine Learning for Business Analytics

---

## 1. Problem Statement

**Can we predict whether the home team will win an NBA game using only pre-game information?**

This is a **binary classification** problem:
- **Target variable:** `HOME_TEAM_WINS` (1 = home win, 0 = away win)
- **Predictors:** Team rankings, rolling performance stats, conference matchups, and season context — all known *before* tipoff

**Why it matters:** Accurate game outcome predictions have applications in sports analytics, broadcast planning, fantasy sports, and sports betting markets.

---

## 2. Data Sources

All data comes from the [NBA dataset on GitHub](https://github.com/blaisescileppi/ML-Business-Analytics-Project/tree/main/data-ba576project), consisting of 5 CSV files:

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `games.csv` | 26,651 | 21 | Game-level stats (points, shooting %, assists, rebounds) for home and away teams |
| `games_details.csv` | 668,628 | 29 | Player-level box scores per game |
| `players.csv` | 7,228 | 4 | Player name, team ID, player ID, season |
| `ranking.csv` | 210,342 | 13 | Team standings over time (W/L record, W_PCT, conference, home/road records) |
| `teams.csv` | 30 | 14 | Team metadata (arena, city, owner, coach) |

**Coverage:** 20 NBA seasons (2003–2022), 30 teams.

---

## 3. Data Cleaning

### Games Table
- Converted `GAME_DATE_EST` to datetime
- Dropped rows with missing game stats (unplayed games): 26,651 → **26,552 games**
- Cast `HOME_TEAM_WINS` to integer

### Rankings Table
- Converted `STANDINGSDATE` to datetime
- Parsed `HOME_RECORD` and `ROAD_RECORD` strings (e.g., "10-3") into separate numeric columns: `HOME_W`, `HOME_L`, `ROAD_W`, `ROAD_L`

### Handling Missing Values
- Rolling averages require a minimum of 3 games (`min_periods=3`), so early-season games are dropped
- Rankings merge uses `merge_asof` (backward direction), so games before the first standings update are dropped
- Final dataset after cleaning: **26,639 games**

---

## 4. Feature Engineering

### Design Principle
We only use features available **before** the game starts. Actual game stats (points scored, FG%, etc.) would constitute data leakage.

### 29 Features in 4 Categories

#### A. Season Rankings (8 features)
Merged from `ranking.csv` using `merge_asof` to get the most recent standings *before* each game date:
- `HOME_W_PCT` / `AWAY_W_PCT` — overall win percentages
- `W_PCT_DIFF` — home minus away win percentage
- `HOME_HOME_W_PCT` — home team's win rate at home specifically
- `AWAY_ROAD_W_PCT` — away team's win rate on the road specifically
- `HOME_VS_AWAY_ROAD` — differential of home-at-home vs away-on-road
- `HOME_G` / `AWAY_G` — games played (proxy for season progress)

#### B. Rolling Performance Averages (18 features)
For each team, computed **10-game rolling averages** (shifted by 1 to prevent leakage) of:
- Points (`PTS`), Field Goal % (`FG_PCT`), Free Throw % (`FT_PCT`), 3-Point % (`FG3_PCT`), Assists (`AST`), Rebounds (`REB`)

This yields 6 home rolling + 6 away rolling + 6 differential features = 18 total.

#### C. Matchup Context (1 feature)
- `SAME_CONFERENCE` — binary indicator for intra-conference game

#### D. Calendar Features (2 features)
- `MONTH` — month of game
- `DAY_OF_WEEK` — day of week (0=Monday, 6=Sunday)

---

## 5. Exploratory Data Analysis

### Key Findings

1. **Home court advantage is real but declining:** The overall home win rate is 58.9%, but it has decreased in recent seasons (especially during COVID's 2020 bubble season)

2. **Win % differential strongly separates outcomes:** When the home team has a much higher win percentage, they win ~80%+ of the time. When much lower, ~30%.

3. **Feature correlations with target:**
   - `W_PCT_DIFF` is the strongest single predictor (r ≈ 0.35)
   - `HOME_VS_AWAY_ROAD` and `ROLL_PTS_DIFF` are also strong
   - `SAME_CONFERENCE` has near-zero correlation (conference doesn't predict outcome)

4. **Monthly patterns:** Home advantage is slightly stronger in early season (October–November)

### Plots Generated
- `01_target_distribution.png` — Target balance and home win rate by season
- `02_feature_analysis.png` — Feature distributions and win rates by subgroup
- `03_correlation_heatmap.png` — Feature correlation matrix
- `04_winrate_by_quality.png` — Home win rate vs. team quality differential

---

## 6. Modeling

### Train/Test Split
- **80/20 stratified split** (preserving target class balance)
- Training: 21,311 games | Test: 5,328 games
- Features standardized with `StandardScaler` for logistic regression models

### Baseline
- **Majority class classifier:** Always predict home win → **58.89% accuracy**

### Models Trained

All models tuned via **5-fold cross-validation** using `GridSearchCV`:

| # | Model | Hyperparameters Tuned | Best Params | CV Accuracy | Test Accuracy | Test AUC |
|---|-------|-----------------------|-------------|-------------|---------------|----------|
| 1 | Logistic Regression | (default) | — | 0.7498 | 0.7599 | 0.8343 |
| 2 | **Ridge Logistic (L2)** | C: [0.01, 0.1, 1, 10] | **C=0.01** | **0.7509** | **0.7618** | **0.8345** |
| 3 | Lasso Logistic (L1) | C: [0.01, 0.1, 1, 10] | C=0.1 | 0.7505 | 0.7601 | 0.8345 |
| 4 | Decision Tree | max_depth, min_samples_leaf | depth=5, leaf=20 | 0.7454 | 0.7500 | 0.8311 |
| 5 | Random Forest | n_estimators, max_depth, min_samples_leaf | 300 trees, depth=10, leaf=20 | 0.7514 | 0.7577 | 0.8358 |
| 6 | Gradient Boosting | n_estimators, max_depth, learning_rate | 100 trees, depth=3, lr=0.05 | 0.7498 | 0.7592 | **0.8377** |
| 7 | XGBoost | n_estimators, max_depth, learning_rate | 100 trees, depth=5, lr=0.1 | 0.7498 | 0.7575 | 0.8353 |

### Best Model: Ridge Logistic Regression (L2)

**Test Accuracy: 76.18% | AUC: 0.8345**

```
Classification Report:
              precision    recall  f1-score   support
    Away Win       0.74      0.65      0.69      2191
    Home Win       0.78      0.84      0.81      3137
    accuracy                           0.76      5328
```

### Key Observations
- **All 7 models beat the baseline** by 16–17 percentage points
- Ridge logistic regression performed best by test accuracy (76.2%)
- Gradient Boosting had the highest AUC (0.838), indicating superior ranking ability
- Models are remarkably close in performance (~1.2% spread), suggesting the signal ceiling is near 76%
- The Lasso model (L1) drove some feature coefficients toward zero, confirming some features add minimal value
- Decision Tree was the weakest, as expected for a non-ensemble method

---

## 7. Parameter Tuning Analysis

### Ridge Regression: C Parameter
- Small C (strong regularization) performs best (C=0.01)
- Accuracy degrades slightly with very weak regularization, indicating mild overfitting risk

### Random Forest: Number of Trees
- Performance improves from 10 to 100 trees, then plateaus
- 300 trees offers marginal improvement over 100 with 3x compute cost

### Gradient Boosting: Learning Rate × n_estimators
- Lower learning rates (0.01, 0.05) with moderate tree counts perform well
- Higher learning rate (0.1) risks overfitting with many trees

---

## 8. Feature Importance

Top features by absolute coefficient (Ridge Logistic Regression):
1. `W_PCT_DIFF` — Season win percentage differential
2. `HOME_VS_AWAY_ROAD` — Home team's home record vs away team's road record
3. `ROLL_PTS_DIFF` — Recent scoring differential
4. `HOME_W_PCT` — Home team's overall win percentage
5. `ROLL_FG_PCT_DIFF` — Recent field goal percentage differential

---

## 9. Limitations and Future Work

### Current Limitations
- **No injury data:** Player availability is a major factor in NBA outcomes but isn't captured
- **No rest/fatigue:** Back-to-back games significantly affect performance
- **No head-to-head history:** Some teams consistently perform well against specific opponents
- **Early-season sparsity:** Rolling averages are unreliable in the first weeks of a season
- **Inherent randomness:** NBA games have high variance; even Vegas lines predict only ~68-70% of outcomes

### Potential Improvements
- Incorporate player-level features from `games_details.csv` (aggregate star player stats)
- Add rest days between games as a feature
- Include historical head-to-head matchup records
- Try neural network approaches or stacking ensembles
- Use the 2020 COVID bubble season as a natural experiment (no real home court)

---

## 10. File Structure

```
Project/
├── data/
│   ├── games.csv
│   ├── games_details.csv
│   ├── players.csv
│   ├── ranking.csv
│   └── teams.csv
├── plots/
│   ├── 01_target_distribution.png
│   ├── 02_feature_analysis.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_winrate_by_quality.png
│   ├── 05_model_comparison.png
│   ├── 06_confusion_matrix.png
│   ├── 07_feature_importance.png
│   ├── 08_tuning_effects.png
│   └── 09_gb_tuning_heatmap.png
├── nba_project.ipynb          # Main notebook (run this)
├── run_pipeline.py            # Standalone script version
├── model_results.csv          # Exported model comparison table
├── project.pdf                # Assignment description
└── PROJECT_DOCUMENTATION.md   # This file
```

---

## 11. How to Run

```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter ipykernel

# Install Jupyter kernel
python3 -m ipykernel install --user --name=ba576 --display-name="BA576 Project"

# Option 1: Run the notebook interactively
jupyter notebook nba_project.ipynb

# Option 2: Run the standalone script
python3 run_pipeline.py
```

**Note:** XGBoost requires the OpenMP runtime. On macOS, install with `brew install libomp`.
