"""HIT140 Objective 2: leakage-aware first-pass modelling from FBref results only.
Source: user-exported FBref 2026 World Cup Scores & Fixtures.
No external historical FIFA ranking, xG, shots or other data are invented.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'input'
OUTPUT = ROOT
(OUTPUT / 'figures').mkdir(parents=True, exist_ok=True)
MATCH_FILE = SOURCE / 'objective2_match_base_104.csv'
TEAM_FILE = SOURCE / 'objective2_team_match_base_208.csv'

matches = pd.read_csv(MATCH_FILE).sort_values(['date','match_id']).reset_index(drop=True)
teams = pd.read_csv(TEAM_FILE).sort_values(['date','match_id','listed_side']).reset_index(drop=True)
assert len(matches)==104 and len(teams)==208
assert matches.match_id.nunique()==104 and teams.groupby('match_id').size().eq(2).all()
assert (matches.team_a_goals-matches.team_b_goals==matches.goal_difference).all()
assert teams.groupby('match_id').goals_scored.sum().sort_index().equals((matches.set_index('match_id').team_a_goals+matches.set_index('match_id').team_b_goals).sort_index())

# Compute historical features one match at a time; update each team's history ONLY
# after both sides' pre-match features have been recorded.
history = {}
records = []
for m in matches.itertuples(index=False):
    date = pd.Timestamp(m.date)
    context = {}
    for side, name, opponent, goals_for, goals_against in [
        ('A', m.team_a, m.team_b, m.team_a_goals, m.team_b_goals),
        ('B', m.team_b, m.team_a, m.team_b_goals, m.team_a_goals)]:
        prior = history.get(name, [])
        n = len(prior)
        gf = np.array([p['gf'] for p in prior], dtype=float)
        ga = np.array([p['ga'] for p in prior], dtype=float)
        last_date = prior[-1]['date'] if n else pd.NaT
        rest = (date-last_date).days if n else np.nan
        context[side] = dict(
            match_id=m.match_id, date=m.date, round=m.round, team=name,
            opponent=opponent, listed_side=side, goals_scored=goals_for,
            goals_conceded=goals_against, prior_matches=n,
            prior_avg_gf=float(gf.mean()) if n else np.nan,
            prior_avg_ga=float(ga.mean()) if n else np.nan,
            prior_avg_gd=float((gf-ga).mean()) if n else np.nan,
            prior_win_rate=float((gf>ga).mean()) if n else np.nan,
            prior_draw_rate=float((gf==ga).mean()) if n else np.nan,
            prior_clean_sheet_rate=float((ga==0).mean()) if n else np.nan,
            prior_points_per_match=float(np.where(gf>ga,3,np.where(gf==ga,1,0)).mean()) if n else np.nan,
            prior_scoring_rate=float((gf>0).mean()) if n else np.nan,
            prior_rest_days=rest,
            knockout_indicator=int(m.round!='Group stage')
        )
    records.extend([context['A'], context['B']])
    history.setdefault(m.team_a, []).append(dict(date=date,gf=m.team_a_goals,ga=m.team_b_goals))
    history.setdefault(m.team_b, []).append(dict(date=date,gf=m.team_b_goals,ga=m.team_a_goals))

pre = pd.DataFrame(records)
a = pre[pre.listed_side=='A'].set_index('match_id')
b = pre[pre.listed_side=='B'].set_index('match_id')

# 2.1 eight features (match-level relative features)
f21 = [
    'avg_goal_difference_diff', 'win_rate_diff', 'clean_sheet_rate_diff',
    'draw_rate_diff', 'rest_days_diff', 'cohost_side_indicator',
    'avg_goals_conceded_diff', 'knockout_indicator'
]
d21 = matches[['match_id','date','round','team_a','team_b','goal_difference']].set_index('match_id')
for col, source in [
    ('avg_goal_difference_diff','prior_avg_gd'),
    ('win_rate_diff','prior_win_rate'),
    ('clean_sheet_rate_diff','prior_clean_sheet_rate'),
    ('draw_rate_diff','prior_draw_rate'),
    ('rest_days_diff','prior_rest_days'),
    ('avg_goals_conceded_diff','prior_avg_ga')]:
    d21[col] = a[source] - b[source]
# Cohost indicator is known before kickoff: +1 listed A is cohost,
# -1 listed B is cohost, 0 neither/both. It is not a measured home advantage.
cohosts = {'Canada', 'Mexico', 'USA'}
d21['cohost_side_indicator'] = (d21.team_a.isin(cohosts).astype(int)
                                      - d21.team_b.isin(cohosts).astype(int))
d21['knockout_indicator'] = a['knockout_indicator']
d21 = d21.reset_index().sort_values(['date','match_id']).reset_index(drop=True)

# 2.2 eight features (team-match level). Conceptual overlap is only
# win rate, goals conceded, rest days and knockout stage = four.
f22 = [
    'team_avg_goals_scored', 'opponent_avg_goals_scored', 'team_win_rate',
    'opponent_avg_goals_conceded', 'team_points_per_match',
    'team_scoring_rate', 'team_rest_days', 'knockout_indicator'
]
d22 = pre[['match_id','date','round','team','opponent','listed_side','goals_scored']].copy()
pre_idx = pre.set_index(['match_id','team'])
for col, source in [
    ('team_avg_goals_scored','prior_avg_gf'),
    ('team_win_rate','prior_win_rate'),
    ('team_points_per_match','prior_points_per_match'),
    ('team_scoring_rate','prior_scoring_rate'),
    ('team_rest_days','prior_rest_days'),
    ('knockout_indicator','knockout_indicator')]:
    d22[col] = pre[source].to_numpy()
for col, source in [
    ('opponent_avg_goals_scored','prior_avg_gf'),
    ('opponent_avg_goals_conceded','prior_avg_ga')]:
    d22[col] = [pre_idx.loc[(r.match_id,r.opponent),source] for r in d22.itertuples(index=False)]
d22 = d22.sort_values(['date','match_id','listed_side']).reset_index(drop=True)

# An alternative for 2.1, tested on training-only cross-validation:
# total prior appearances (a stage/experience proxy, not a future statistic).
d21['candidate_prior_matches_sum'] = (a['prior_matches'] + b['prior_matches']).to_numpy()
# Alternative 2.2: historical fraction of matches in which opponent scored.
opp_scoring = pre_idx['prior_scoring_rate']
d22['candidate_opponent_scoring_rate'] = [opp_scoring.loc[(r.match_id,r.opponent)]
                                           for r in d22.itertuples(index=False)]
assert len(d21)==104 and len(d22)==208
assert d21.match_id.nunique()==104 and d22.groupby('match_id').size().eq(2).all()
assert len(f21)==len(f22)==8
assert not np.isinf(d21[f21].to_numpy(dtype=float)).any()
assert not np.isinf(d22[f22].to_numpy(dtype=float)).any()
assert d21['knockout_indicator'].isin([0,1]).all()
assert d22['knockout_indicator'].isin([0,1]).all()
# Each team has zero historical matches in its first game; never impute using future games.
assert pre.groupby('team').first()['prior_matches'].eq(0).all()


# Chronological 80/20 split BY MATCH. Both 2.2 rows for a match remain together.
match_ids = matches.match_id.tolist()
cut = int(len(match_ids)*0.8)
train_ids, test_ids = set(match_ids[:cut]), set(match_ids[cut:])
assert len(train_ids)==83 and len(test_ids)==21

# Time-ordered folds over unique match IDs, never splitting the two sides.
def chronological_cv_rows(frame, n_splits=4):
    unique_ids = [x for x in match_ids if x in set(frame.match_id)]
    n = len(unique_ids)
    boundaries = np.linspace(int(n*0.30), n, n_splits+1, dtype=int)
    for i in range(n_splits):
        train_match_ids = set(unique_ids[:boundaries[i]])
        valid_match_ids = set(unique_ids[boundaries[i]:boundaries[i+1]])
        tr = np.flatnonzero(frame.match_id.isin(train_match_ids).to_numpy())
        va = np.flatnonzero(frame.match_id.isin(valid_match_ids).to_numpy())
        if len(tr)>0 and len(va)>0:
            yield tr, va

def model_pipeline(estimator):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('regression', estimator)])

# Training-only experiments. Never inspect test RMSE to choose predictors.
def candidate_cv(frame, features, target, model):
    train = frame[frame.match_id.isin(train_ids)].copy().reset_index(drop=True)
    errors=[]
    for tr, va in chronological_cv_rows(train):
        m=clone(model)
        m.fit(train.iloc[tr][features], train.iloc[tr][target])
        pred=m.predict(train.iloc[va][features])
        errors.append(float(np.sqrt(mean_squared_error(train.iloc[va][target],pred))))
    return float(np.mean(errors))

ridge_100=model_pipeline(Ridge(alpha=100))
experiment_rows=[]
# Keep the 2.1 specification to exactly eight explanatory variables.
for label, column in [('Cohost side indicator','cohost_side_indicator'),
                      ('Sum of prior matches','candidate_prior_matches_sum')]:
    feats=[x for x in f21 if x!='cohost_side_indicator']+[column]
    cv=candidate_cv(d21,feats,'goal_difference',ridge_100)
    experiment_rows.append(dict(task='regression21',experiment=label,cv_RMSE=cv))
# Choice is based solely on training CV; tie-break by simpler cohost indicator.
selected21=min((r for r in experiment_rows if r['task']=='regression21'),key=lambda r:r['cv_RMSE'])
if selected21['experiment']=='Sum of prior matches':
    f21=[x if x!='cohost_side_indicator' else 'candidate_prior_matches_sum' for x in f21]

for label, column in [('Points per match','team_points_per_match'),
                      ('Opponent scoring rate','candidate_opponent_scoring_rate')]:
    feats=[x for x in f22 if x!='team_points_per_match']+[column]
    cv=candidate_cv(d22,feats,'goals_scored',ridge_100)
    experiment_rows.append(dict(task='regression22',experiment=label,cv_RMSE=cv))
selected22=min((r for r in experiment_rows if r['task']=='regression22'),key=lambda r:r['cv_RMSE'])
if selected22['experiment']=='Opponent scoring rate':
    f22=[x if x!='team_points_per_match' else 'candidate_opponent_scoring_rate' for x in f22]

pd.DataFrame(experiment_rows).to_csv(OUTPUT/'predictor_experiments_training_cv.csv',index=False)
# Drop candidate columns not selected: exactly eight explanatory columns remain.
d21=d21[['match_id','date','round','team_a','team_b','goal_difference']+f21]
d22=d22[['match_id','date','round','team','opponent','listed_side','goals_scored']+f22]
assert len(f21)==len(f22)==8
assert all(d21[f21].nunique(dropna=True)>1), 'A predictor is constant'
assert all(d22[f22].nunique(dropna=True)>1), 'A predictor is constant'
d21.to_csv(OUTPUT/'regression21_104_eight_predictors.csv',index=False)
d22.to_csv(OUTPUT/'regression22_208_eight_predictors.csv',index=False)

def scores(y, pred):
    return dict(MAE=float(mean_absolute_error(y,pred)),
                RMSE=float(np.sqrt(mean_squared_error(y,pred))),
                R2=float(r2_score(y,pred)))

def evaluate(label, df, feature_names, target):
    train = df[df.match_id.isin(train_ids)].copy().reset_index(drop=True)
    test = df[df.match_id.isin(test_ids)].copy().reset_index(drop=True)
    assert set(train.match_id).isdisjoint(test.match_id)
    X_train, y_train = train[feature_names], train[target]
    X_test, y_test = test[feature_names], test[target]
    specs = {
        'Mean baseline': DummyRegressor(strategy='mean'),
        'OLS linear regression': model_pipeline(LinearRegression()),
        'Ridge alpha=0.1': model_pipeline(Ridge(alpha=0.1)),
        'Ridge alpha=1': model_pipeline(Ridge(alpha=1.0)),
        'Ridge alpha=10': model_pipeline(Ridge(alpha=10.0)),
        'Ridge alpha=100': model_pipeline(Ridge(alpha=100.0)),
        'Lasso alpha=0.01': model_pipeline(Lasso(alpha=0.01, max_iter=20000)),
        'Lasso alpha=0.1': model_pipeline(Lasso(alpha=0.1, max_iter=20000)),
    }
    rows = []
    fitted = {}
    for name, estimator in specs.items():
        fold_errors=[]
        for idx_train, idx_valid in chronological_cv_rows(train):
            m = clone(estimator)
            m.fit(X_train.iloc[idx_train], y_train.iloc[idx_train])
            fold_errors.append(float(np.sqrt(mean_squared_error(y_train.iloc[idx_valid],m.predict(X_train.iloc[idx_valid])))))
        model = clone(estimator).fit(X_train,y_train)
        fitted[name] = model
        train_metrics=scores(y_train, model.predict(X_train))
        test_metrics=scores(y_test, model.predict(X_test))
        rows.append({'task':label, 'model':name,
                     'cv_RMSE':float(np.mean(fold_errors)),
                     'train_RMSE':train_metrics['RMSE'],
                     'test_MAE':test_metrics['MAE'],
                     'test_RMSE':test_metrics['RMSE'],
                     'test_R2':test_metrics['R2']})
    result = pd.DataFrame(rows).sort_values('cv_RMSE').reset_index(drop=True)
    best_name = result.iloc[0]['model']  # selected on TRAIN-only CV
    selected = fitted[best_name]
    ols = fitted['OLS linear regression']
    ols_train_r2 = r2_score(y_train, ols.predict(X_train))
    n, p = len(train), len(feature_names)
    ols_adj_r2 = float(1-(1-ols_train_r2)*(n-1)/(n-p-1))
    predictions = test[['match_id','date',target]].copy()
    predictions['predicted'] = selected.predict(X_test)
    predictions['residual'] = predictions[target] - predictions['predicted']
    predictions.to_csv(OUTPUT/f'{label}_holdout_predictions.csv',index=False)
    result.to_csv(OUTPUT/f'{label}_model_comparison.csv',index=False)
    fig, ax = plt.subplots(figsize=(6.3,4.2))
    ax.scatter(predictions[target],predictions.predicted, alpha=.7)
    lo=min(predictions[target].min(), predictions.predicted.min())
    hi=max(predictions[target].max(), predictions.predicted.max())
    ax.plot([lo,hi],[lo,hi],linestyle='--',linewidth=1)
    ax.set(xlabel=f'Actual {target.replace("_"," ")}', ylabel='Predicted',
           title=f'{label}: holdout actual vs predicted ({best_name})')
    fig.tight_layout();fig.savefig(OUTPUT/'figures'/f'{label}_actual_vs_predicted.png',dpi=170);plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.3,4.2))
    ax.scatter(predictions.predicted,predictions.residual,alpha=.7)
    ax.axhline(0,linestyle='--',linewidth=1)
    ax.set(xlabel='Predicted',ylabel='Actual − predicted',title=f'{label}: holdout residuals')
    fig.tight_layout();fig.savefig(OUTPUT/'figures'/f'{label}_residuals.png',dpi=170);plt.close(fig)
    return result, {'selected_by_cv':best_name, 'train_matches':len(train_ids),
                    'test_matches':len(test_ids), 'train_rows':len(train), 'test_rows':len(test),
                    'ols_training_adjusted_R2':ols_adj_r2,
                    'missing_train_cells':int(X_train.isna().sum().sum()),
                    'missing_test_cells':int(X_test.isna().sum().sum())}

results21, info21 = evaluate('regression21',d21,f21,'goal_difference')
results22, info22 = evaluate('regression22',d22,f22,'goals_scored')
info = {
    'source':'Student FBref 2026 World Cup Scores & Fixtures export; no other stats sourced',
    'predictor_overlap_concepts':['win rate','goals conceded','rest days','knockout stage'],
    'missing_values':'Historical features before a team first plays are NA, imputed using training-fold median in sklearn pipeline; zero match-count is factual.',
    'data_caution':'Only prior 2026 tournament match results used. No pre-tournament team strength or shooting/xG data. Strong limitations for first-round prediction and generalisation.',
    'regression21':info21, 'regression22':info22,
    'predictors21':f21, 'predictors22':f22,
    'training_only_feature_experiments':experiment_rows,
    'selected_feature_21':selected21['experiment'],
    'selected_feature_22':selected22['experiment']
}
(OUTPUT/'run_summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print('Reg 2.1', d21.shape, 'Reg 2.2',d22.shape)
print('2.1 best by chronological CV:', info21['selected_by_cv'])
print(results21[['model','cv_RMSE','test_RMSE','test_R2']].round(3).to_string(index=False))
print('2.2 best by chronological CV:', info22['selected_by_cv'])
print(results22[['model','cv_RMSE','test_RMSE','test_R2']].round(3).to_string(index=False))

# Supplemental diagnostics: TRAINING correlations; HOLDOUT residual shape.
from scipy import stats
for label, frame, features, target in [
    ('regression21', d21, f21, 'goal_difference'),
    ('regression22', d22, f22, 'goals_scored')]:
    train = frame[frame.match_id.isin(train_ids)].copy()
    train[features+[target]].describe().to_csv(OUTPUT/f'{label}_training_descriptives.csv')
    corr=train[features].corr()
    corr.to_csv(OUTPUT/f'{label}_training_correlations.csv')
    fig,ax=plt.subplots(figsize=(8,6.5))
    im=ax.imshow(corr.fillna(0),vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(features)));ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features,rotation=90,fontsize=8)
    ax.set_yticklabels(features,fontsize=8)
    fig.colorbar(im,ax=ax)
    ax.set_title(f'{label}: training predictor correlations')
    fig.tight_layout();fig.savefig(OUTPUT/'figures'/f'{label}_correlation.png',dpi=170,bbox_inches='tight');plt.close(fig)
    residuals=pd.read_csv(OUTPUT/f'{label}_holdout_predictions.csv')['residual'].dropna()
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].hist(residuals,bins=10,edgecolor='black')
    axes[0].set(xlabel='Actual minus predicted (goals)',title='Holdout residual distribution')
    stats.probplot(residuals,dist='norm',plot=axes[1]);axes[1].set_title('Holdout residual Q-Q plot')
    fig.tight_layout();fig.savefig(OUTPUT/'figures'/f'{label}_diagnostics.png',dpi=170);plt.close(fig)
    print(label,'maximum absolute training feature correlation:',
          round(float(corr.where(~np.eye(len(corr),dtype=bool)).abs().max().max()),3))
