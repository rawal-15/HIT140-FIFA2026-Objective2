# FIFA World Cup 2026 — HIT140 Objective 2

Reproducible Python analysis for HIT140 Foundations of Data Science, Assessment 3 (Objective 2).

## Research tasks

- **Regression 2.1:** predict goal difference (Team A minus Team B) across 104 matches.
- **Regression 2.2:** predict goals scored by a team, using 208 team–match observations.

Both tasks use eight pre-match explanatory variables. Historical features are calculated from matches played before the relevant kickoff; the project compares a mean baseline, ordinary least squares, Ridge and Lasso with chronological validation. Results include model diagnostics and limitations.

## Repository files

- `input/worldcup2026_matches_raw.csv`: source-exported match table.
- `input/prepare_objective2.py`: prepare 104-match and 208-team-match base datasets.
- `input/objective2_match_base_104.csv` and `input/objective2_team_match_base_208.csv`: prepared base data.
- `run_objective2.py`: reproducible feature engineering, model selection, evaluation and plots.
- `FIFA2026_Objective2_Final_Analysis.ipynb`: notebook version of the analysis.
- `regression21_104_eight_predictors.csv` and `regression22_208_eight_predictors.csv`: modelling datasets.
- `regression*_model_comparison.csv`, `regression*_holdout_predictions.csv`, `regression*_training_correlations.csv`, `regression*_training_descriptives.csv`: analysis outputs.
- `predictor_experiments_training_cv.csv`: training-validation feature experiments.
- `figures/`: eight diagnostic and evaluation figures.
- `run_summary.json`: machine-readable run summary.
- `DATA_SOURCE_SPOT_CHECKS.md`: limited source cross-check notes.
- `requirements.txt`: Python dependencies.
- `PUSH_TO_GITHUB_WINDOWS.txt`: optional upload instructions.

## Reproduce on Windows using VS Code

Open the project root folder in VS Code. In **Terminal > New Terminal**, run:

```powershell
py -m pip install -r requirements.txt
py input/prepare_objective2.py
py run_objective2.py
```

The scripts recreate derived datasets and plots in this folder. Alternatively, open the `.ipynb` notebook and run all cells with the project root as the working directory.

## Data provenance and limitations

The source match table was exported by a student from FBref's 2026 World Cup scores and fixtures pages: https://fbref.com/en/ . The seven checks in `DATA_SOURCE_SPOT_CHECKS.md` are spot checks only; they do not independently verify all 104 matches. Match results, assumptions, and model outputs should be reviewed against the source before assessment submission.

The 104/208 sample design is specific to this tournament. First-match historical predictors are missing by design and imputed within training folds. The goal-difference model has weak holdout performance, and the goals-scored model provides only limited predictive improvement. These results should not be presented as reliable match forecasts.

## Academic integrity and contributions

ChatGPT assisted with Python code development, analysis diagnostics and documentation drafting. Group members are responsible for verifying the results, understanding the work, recording their actual contributions and completing any required university AI-use declaration. GitHub upload history is not itself evidence that every member wrote code.

Private student details, signatures, group report and Turnitin materials are not part of this public code package.
