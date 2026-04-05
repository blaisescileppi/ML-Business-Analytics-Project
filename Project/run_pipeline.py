"""
NBA Home Team Win Prediction - Full Pipeline Script
Runs the entire ML pipeline and prints results.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier

sns.set_style('whitegrid')
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

DATA_DIR = 'data' if os.path.exists('data/games.csv') else '../data-ba576project'

games = pd.read_csv(f'{DATA_DIR}/games.csv')
details = pd.read_csv(f'{DATA_DIR}/games_details.csv')
players = pd.read_csv(f'{DATA_DIR}/players.csv')
ranking = pd.read_csv(f'{DATA_DIR}/ranking.csv')
teams = pd.read_csv(f'{DATA_DIR}/teams.csv')

for name, df in [('games', games), ('details', details), ('players', players),
                 ('ranking', ranking), ('teams', teams)]:
    print(f'  {name:>15s}: {df.shape[0]:>7,} rows x {df.shape[1]:>2} cols')

print("\n" + "=" * 60)
print("STEP 2: Cleaning data")
print("=" * 60)

games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
games_clean = games.dropna(subset=['PTS_home', 'PTS_away', 'HOME_TEAM_WINS']).copy()
games_clean['HOME_TEAM_WINS'] = games_clean['HOME_TEAM_WINS'].astype(int)
print(f'Games after cleaning: {len(games_clean):,}')
print(f'Home win rate: {games_clean["HOME_TEAM_WINS"].mean():.3f}')
print(f'Seasons: {sorted(games_clean["SEASON"].unique())}')

ranking['STANDINGSDATE'] = pd.to_datetime(ranking['STANDINGSDATE'])
ranking[['HOME_W', 'HOME_L']] = ranking['HOME_RECORD'].str.split('-', expand=True).astype(float)
ranking[['ROAD_W', 'ROAD_L']] = ranking['ROAD_RECORD'].str.split('-', expand=True).astype(float)

print("\n" + "=" * 60)
print("STEP 3: Feature engineering")
print("=" * 60)

ranking_cols = ['TEAM_ID', 'STANDINGSDATE', 'CONFERENCE', 'G', 'W', 'L', 'W_PCT',
                'HOME_W', 'HOME_L', 'ROAD_W', 'ROAD_L']
rank_slim = ranking[ranking_cols].copy()
games_sorted = games_clean.sort_values('GAME_DATE_EST').copy()
rank_sorted = rank_slim.sort_values('STANDINGSDATE').copy()

home_rank = pd.merge_asof(
    games_sorted[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID']].rename(columns={'HOME_TEAM_ID': 'TEAM_ID'}),
    rank_sorted, left_on='GAME_DATE_EST', right_on='STANDINGSDATE', by='TEAM_ID', direction='backward'
)
home_rank = home_rank.rename(columns={
    'CONFERENCE': 'HOME_CONF', 'G': 'HOME_G', 'W': 'HOME_W_season', 'L': 'HOME_L_season',
    'W_PCT': 'HOME_W_PCT', 'HOME_W': 'HOME_HOME_W', 'HOME_L': 'HOME_HOME_L',
    'ROAD_W': 'HOME_ROAD_W', 'ROAD_L': 'HOME_ROAD_L'
}).drop(columns=['STANDINGSDATE', 'TEAM_ID'])

away_rank = pd.merge_asof(
    games_sorted[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID']].rename(columns={'VISITOR_TEAM_ID': 'TEAM_ID'}),
    rank_sorted, left_on='GAME_DATE_EST', right_on='STANDINGSDATE', by='TEAM_ID', direction='backward'
)
away_rank = away_rank.rename(columns={
    'CONFERENCE': 'AWAY_CONF', 'G': 'AWAY_G', 'W': 'AWAY_W_season', 'L': 'AWAY_L_season',
    'W_PCT': 'AWAY_W_PCT', 'HOME_W': 'AWAY_HOME_W', 'HOME_L': 'AWAY_HOME_L',
    'ROAD_W': 'AWAY_ROAD_W', 'ROAD_L': 'AWAY_ROAD_L'
}).drop(columns=['STANDINGSDATE', 'TEAM_ID'])
print(f'Rankings merged: home={home_rank.shape}, away={away_rank.shape}')

stat_cols = ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']
home_games = games_sorted[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'SEASON'] +
                          [c + '_home' for c in stat_cols]].copy()
home_games.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'SEASON'] + stat_cols

away_games = games_sorted[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'SEASON'] +
                          [c + '_away' for c in stat_cols]].copy()
away_games.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'SEASON'] + stat_cols

all_team_games = pd.concat([home_games, away_games]).sort_values(['TEAM_ID', 'GAME_DATE_EST'])

WINDOW = 10
rolling = (
    all_team_games.groupby('TEAM_ID')[stat_cols]
    .apply(lambda x: x.shift(1).rolling(WINDOW, min_periods=3).mean())
)
rolling.columns = [f'ROLL_{c}' for c in stat_cols]
all_team_games = pd.concat([all_team_games.reset_index(drop=True), rolling.reset_index(drop=True)], axis=1)

roll_cols = [f'ROLL_{c}' for c in stat_cols]
home_rolling = all_team_games.loc[
    all_team_games['GAME_ID'].isin(home_games['GAME_ID'])
].groupby('GAME_ID').first()[roll_cols].rename(columns={c: f'HOME_{c}' for c in roll_cols})

away_rolling = all_team_games.loc[
    all_team_games['GAME_ID'].isin(away_games['GAME_ID'])
].groupby('GAME_ID').last()[roll_cols].rename(columns={c: f'AWAY_{c}' for c in roll_cols})
print(f'Rolling features: home={home_rolling.shape}, away={away_rolling.shape}')

df = games_sorted[['GAME_ID', 'GAME_DATE_EST', 'SEASON', 'HOME_TEAM_ID',
                    'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']].copy()
df = df.merge(home_rank, on=['GAME_ID', 'GAME_DATE_EST'], how='left')
df = df.merge(away_rank, on=['GAME_ID', 'GAME_DATE_EST'], how='left')
df = df.merge(home_rolling, on='GAME_ID', how='left')
df = df.merge(away_rolling, on='GAME_ID', how='left')

df['W_PCT_DIFF'] = df['HOME_W_PCT'] - df['AWAY_W_PCT']
df['HOME_HOME_W_PCT'] = df['HOME_HOME_W'] / (df['HOME_HOME_W'] + df['HOME_HOME_L'])
df['AWAY_ROAD_W_PCT'] = df['AWAY_ROAD_W'] / (df['AWAY_ROAD_W'] + df['AWAY_ROAD_L'])
df['HOME_VS_AWAY_ROAD'] = df['HOME_HOME_W_PCT'] - df['AWAY_ROAD_W_PCT']
df['SAME_CONFERENCE'] = (df['HOME_CONF'] == df['AWAY_CONF']).astype(int)
df['MONTH'] = df['GAME_DATE_EST'].dt.month
df['DAY_OF_WEEK'] = df['GAME_DATE_EST'].dt.dayofweek

for s in stat_cols:
    df[f'ROLL_{s}_DIFF'] = df[f'HOME_ROLL_{s}'] - df[f'AWAY_ROLL_{s}']

feature_cols = [
    'HOME_W_PCT', 'AWAY_W_PCT', 'W_PCT_DIFF',
    'HOME_HOME_W_PCT', 'AWAY_ROAD_W_PCT', 'HOME_VS_AWAY_ROAD',
    'HOME_G', 'AWAY_G',
    'SAME_CONFERENCE', 'MONTH', 'DAY_OF_WEEK',
    'HOME_ROLL_PTS', 'HOME_ROLL_FG_PCT', 'HOME_ROLL_FT_PCT',
    'HOME_ROLL_FG3_PCT', 'HOME_ROLL_AST', 'HOME_ROLL_REB',
    'AWAY_ROLL_PTS', 'AWAY_ROLL_FG_PCT', 'AWAY_ROLL_FT_PCT',
    'AWAY_ROLL_FG3_PCT', 'AWAY_ROLL_AST', 'AWAY_ROLL_REB',
    'ROLL_PTS_DIFF', 'ROLL_FG_PCT_DIFF', 'ROLL_FT_PCT_DIFF',
    'ROLL_FG3_PCT_DIFF', 'ROLL_AST_DIFF', 'ROLL_REB_DIFF',
]
target = 'HOME_TEAM_WINS'

model_df = df.dropna(subset=feature_cols + [target]).copy()
print(f'Final dataset: {model_df.shape[0]:,} games x {len(feature_cols)} features')
print(f'Home win rate: {model_df[target].mean():.3f}')

print("\n" + "=" * 60)
print("STEP 4: EDA Plots")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
season_wr = model_df.groupby('SEASON')[target].mean()
axes[0].bar(season_wr.index.astype(str), season_wr.values, color='steelblue')
axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%')
axes[0].set_xlabel('Season'); axes[0].set_ylabel('Home Win Rate')
axes[0].set_title('Home Court Advantage by Season')
axes[0].tick_params(axis='x', rotation=45); axes[0].legend()

model_df[target].value_counts().plot.bar(ax=axes[1], color=['coral', 'steelblue'])
axes[1].set_xlabel('Home Team Wins'); axes[1].set_ylabel('Count')
axes[1].set_title('Target Variable Distribution')
axes[1].set_xticklabels(['Away Win (0)', 'Home Win (1)'], rotation=0)
plt.tight_layout(); plt.savefig('plots/01_target_distribution.png', bbox_inches='tight'); plt.close()
print('  Saved plots/01_target_distribution.png')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for val, label, color in [(1, 'Home Win', 'steelblue'), (0, 'Away Win', 'coral')]:
    subset = model_df[model_df[target] == val]
    axes[0, 0].hist(subset['W_PCT_DIFF'], bins=40, alpha=0.6, label=label, color=color, density=True)
axes[0, 0].set_xlabel('Win % Difference (Home - Away)'); axes[0, 0].set_title('Win % Difference by Outcome'); axes[0, 0].legend()

for val, label, color in [(1, 'Home Win', 'steelblue'), (0, 'Away Win', 'coral')]:
    subset = model_df[model_df[target] == val]
    axes[0, 1].hist(subset['ROLL_PTS_DIFF'], bins=40, alpha=0.6, label=label, color=color, density=True)
axes[0, 1].set_xlabel('Rolling Pts Diff (Home - Away)'); axes[0, 1].set_title('Rolling Points Differential'); axes[0, 1].legend()

month_wr = model_df.groupby('MONTH')[target].mean()
axes[1, 0].bar(month_wr.index, month_wr.values, color='steelblue')
axes[1, 0].axhline(y=model_df[target].mean(), color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('Month'); axes[1, 0].set_title('Home Win Rate by Month')

conf_wr = model_df.groupby('SAME_CONFERENCE')[target].mean()
axes[1, 1].bar(['Inter-Conference', 'Same Conference'], conf_wr.values, color=['coral', 'steelblue'])
axes[1, 1].set_title('Home Win Rate by Conference Matchup')
plt.tight_layout(); plt.savefig('plots/02_feature_analysis.png', bbox_inches='tight'); plt.close()
print('  Saved plots/02_feature_analysis.png')

corr_cols = ['HOME_W_PCT', 'AWAY_W_PCT', 'W_PCT_DIFF', 'HOME_HOME_W_PCT',
             'AWAY_ROAD_W_PCT', 'HOME_VS_AWAY_ROAD', 'ROLL_PTS_DIFF',
             'ROLL_FG_PCT_DIFF', 'ROLL_AST_DIFF', 'ROLL_REB_DIFF',
             'SAME_CONFERENCE', target]
fig, ax = plt.subplots(figsize=(12, 9))
corr = model_df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title('Correlation Matrix of Key Features with Target')
plt.tight_layout(); plt.savefig('plots/03_correlation_heatmap.png', bbox_inches='tight'); plt.close()
print('  Saved plots/03_correlation_heatmap.png')

model_df_plot = model_df.copy()
model_df_plot['W_PCT_DIFF_BIN'] = pd.cut(model_df_plot['W_PCT_DIFF'], bins=10)
bin_wr = model_df_plot.groupby('W_PCT_DIFF_BIN', observed=True)[target].agg(['mean', 'count'])
fig, ax1 = plt.subplots(figsize=(12, 5)); ax2 = ax1.twinx()
x = range(len(bin_wr))
ax1.bar(x, bin_wr['count'], alpha=0.3, color='gray', label='Game Count')
ax2.plot(x, bin_wr['mean'], 'o-', color='steelblue', linewidth=2, markersize=8, label='Home Win Rate')
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Win % Difference Bin'); ax1.set_ylabel('Games', color='gray'); ax2.set_ylabel('Home Win Rate', color='steelblue')
ax1.set_xticks(x); ax1.set_xticklabels([str(b) for b in bin_wr.index], rotation=45, ha='right')
ax1.set_title('Home Win Rate vs. Team Quality Differential')
lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout(); plt.savefig('plots/04_winrate_by_quality.png', bbox_inches='tight'); plt.close()
print('  Saved plots/04_winrate_by_quality.png')

print("\n" + "=" * 60)
print("STEP 5: Train/Test Split")
print("=" * 60)

X = model_df[feature_cols].values
y = model_df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

baseline_acc = max(y_train.mean(), 1 - y_train.mean())
print(f'Training: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Features: {X_train.shape[1]}')
print(f'Baseline (majority class): {baseline_acc:.4f}')

print("\n" + "=" * 60)
print("STEP 6: Training Models")
print("=" * 60)

results = {}

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob) if y_prob is not None else None
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy', n_jobs=1)
    results[name] = {
        'model': model, 'test_acc': acc, 'test_auc': auc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'y_pred': y_pred, 'y_prob': y_prob,
    }
    auc_str = f'{auc:.4f}' if auc else 'N/A'
    print(f'  {name:30s} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f} | Test: {acc:.4f} | AUC: {auc_str}')
    return model

# 1. Logistic Regression
print('\n[1/7] Logistic Regression...')
evaluate_model('Logistic Regression', LogisticRegression(max_iter=1000), X_train_sc, X_test_sc, y_train, y_test)

# 2. Ridge (L2)
print('[2/7] Ridge Logistic Regression...')
ridge_grid = GridSearchCV(
    LogisticRegression(penalty='l2', max_iter=1000),
    param_grid={'C': [0.01, 0.1, 1, 10]}, cv=5, scoring='accuracy', n_jobs=1
)
ridge_grid.fit(X_train_sc, y_train)
print(f'  Best C: {ridge_grid.best_params_["C"]}')
evaluate_model('Ridge Logistic (L2)', ridge_grid.best_estimator_, X_train_sc, X_test_sc, y_train, y_test)

# 3. Lasso (L1)
print('[3/7] Lasso Logistic Regression...')
lasso_grid = GridSearchCV(
    LogisticRegression(penalty='l1', solver='saga', max_iter=2000),
    param_grid={'C': [0.01, 0.1, 1, 10]}, cv=5, scoring='accuracy', n_jobs=1
)
lasso_grid.fit(X_train_sc, y_train)
print(f'  Best C: {lasso_grid.best_params_["C"]}')
evaluate_model('Lasso Logistic (L1)', lasso_grid.best_estimator_, X_train_sc, X_test_sc, y_train, y_test)

# 4. Decision Tree
print('[4/7] Decision Tree...')
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={'max_depth': [3, 5, 10, 15], 'min_samples_leaf': [10, 20, 50]},
    cv=5, scoring='accuracy', n_jobs=1
)
dt_grid.fit(X_train, y_train)
print(f'  Best: {dt_grid.best_params_}')
evaluate_model('Decision Tree', dt_grid.best_estimator_, X_train, X_test, y_train, y_test)

# 5. Random Forest
print('[5/7] Random Forest...')
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [100, 300], 'max_depth': [10, 15], 'min_samples_leaf': [10, 20]},
    cv=5, scoring='accuracy', n_jobs=1
)
rf_grid.fit(X_train, y_train)
print(f'  Best: {rf_grid.best_params_}')
evaluate_model('Random Forest', rf_grid.best_estimator_, X_train, X_test, y_train, y_test)

# 6. Gradient Boosting
print('[6/7] Gradient Boosting...')
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid={'n_estimators': [100, 300], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
    cv=5, scoring='accuracy', n_jobs=1
)
gb_grid.fit(X_train, y_train)
print(f'  Best: {gb_grid.best_params_}')
evaluate_model('Gradient Boosting', gb_grid.best_estimator_, X_train, X_test, y_train, y_test)

# 7. XGBoost
print('[7/7] XGBoost...')
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    param_grid={'n_estimators': [100, 300], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
    cv=5, scoring='accuracy', n_jobs=1
)
xgb_grid.fit(X_train, y_train)
print(f'  Best: {xgb_grid.best_params_}')
evaluate_model('XGBoost', xgb_grid.best_estimator_, X_train, X_test, y_train, y_test)

print("\n" + "=" * 60)
print("STEP 7: Model Comparison")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': results.keys(),
    'CV Accuracy': [r['cv_mean'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()],
    'Test Accuracy': [r['test_acc'] for r in results.values()],
    'Test AUC': [r['test_auc'] for r in results.values()],
}).sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
print(comparison.to_string(index=False))
print(f'\nBaseline: {baseline_acc:.4f}')

best_name = comparison.iloc[0]['Model']
best = results[best_name]
print(f'\nBest: {best_name} — Acc: {best["test_acc"]:.4f}, AUC: {best["test_auc"]:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, best['y_pred'], target_names=['Away Win', 'Home Win']))

print("\n" + "=" * 60)
print("STEP 8: Generating remaining plots")
print("=" * 60)

# Model comparison bar + ROC
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
models_sorted = comparison.sort_values('Test Accuracy')
colors = ['steelblue' if acc > baseline_acc else 'coral' for acc in models_sorted['Test Accuracy']]
axes[0].barh(models_sorted['Model'], models_sorted['Test Accuracy'], color=colors)
axes[0].axvline(x=baseline_acc, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_acc:.3f})')
axes[0].set_xlabel('Test Accuracy'); axes[0].set_title('Model Test Accuracy'); axes[0].legend()
axes[0].set_xlim(0.5, max(models_sorted['Test Accuracy']) + 0.02)

for name, r in results.items():
    if r['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        axes[1].plot(fpr, tpr, label=f"{name} ({r['test_auc']:.3f})")
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].set_title('ROC Curves'); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('plots/05_model_comparison.png', bbox_inches='tight'); plt.close()
print('  Saved plots/05_model_comparison.png')

# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Away Win', 'Home Win'], yticklabels=['Away Win', 'Home Win'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'Confusion Matrix — {best_name}')
plt.tight_layout(); plt.savefig('plots/06_confusion_matrix.png', bbox_inches='tight'); plt.close()
print('  Saved plots/06_confusion_matrix.png')

# Feature importance
best_model = best['model']
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    imp_label = 'Feature Importance'
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
    imp_label = 'Absolute Coefficient'
else:
    importances = None

if importances is not None:
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.tail(15).plot.barh(ax=ax, color='steelblue')
    ax.set_xlabel(imp_label); ax.set_title(f'Top 15 Features — {best_name}')
    plt.tight_layout(); plt.savefig('plots/07_feature_importance.png', bbox_inches='tight'); plt.close()
    print('  Saved plots/07_feature_importance.png')

# Tuning plots
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = []
for C in C_values:
    lr = LogisticRegression(penalty='l2', C=C, max_iter=1000)
    scores = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring='accuracy', n_jobs=1)
    ridge_scores.append(scores.mean())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(C_values, ridge_scores, 'o-', color='steelblue', linewidth=2)
axes[0].set_xscale('log'); axes[0].set_xlabel('C'); axes[0].set_ylabel('CV Accuracy')
axes[0].set_title('Ridge Logistic — Tuning C')

n_est_values = [10, 50, 100, 200, 300]
rf_scores_tune = []
for n in n_est_values:
    rf = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)
    rf_scores_tune.append(scores.mean())

axes[1].plot(n_est_values, rf_scores_tune, 'o-', color='forestgreen', linewidth=2)
axes[1].set_xlabel('Number of Trees'); axes[1].set_ylabel('CV Accuracy')
axes[1].set_title('Random Forest — Effect of n_estimators')
plt.tight_layout(); plt.savefig('plots/08_tuning_effects.png', bbox_inches='tight'); plt.close()
print('  Saved plots/08_tuning_effects.png')

# GB heatmap
lr_values = [0.01, 0.05, 0.1]
n_values = [100, 200, 300]
gb_heatmap = np.zeros((len(lr_values), len(n_values)))
for i, lr_val in enumerate(lr_values):
    for j, n in enumerate(n_values):
        gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr_val, max_depth=3, random_state=42)
        gb_heatmap[i, j] = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1).mean()

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(gb_heatmap, annot=True, fmt='.4f', cmap='YlOrRd', xticklabels=n_values, yticklabels=lr_values, ax=ax)
ax.set_xlabel('n_estimators'); ax.set_ylabel('Learning Rate')
ax.set_title('Gradient Boosting — CV Accuracy Heatmap')
plt.tight_layout(); plt.savefig('plots/09_gb_tuning_heatmap.png', bbox_inches='tight'); plt.close()
print('  Saved plots/09_gb_tuning_heatmap.png')

print("\n" + "=" * 60)
print("DONE! All plots saved to plots/")
print("=" * 60)

# Save results for the notebook
comparison.to_csv('model_results.csv', index=False)
print(f'Results saved to model_results.csv')
