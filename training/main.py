import pandas as pd
import numpy as np
import os
import warnings
from pybaseball import statcast, playerid_lookup 
from sklearn.model_selection import KFold, cross_val_score, train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc 
from datetime import datetime
import pickle 
from sklearn.model_selection import RandomizedSearchCV

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']

DATA_FILE = os.path.join(os.path.expanduser("~"),'UIUC/fa25/CS441/final_projecf/repo/data/pitching_master_data_aggregated.csv')
MODEL_FILE = str(input("Input Model:\n"))
MODEL_FILE = os.path.join(os.path.expanduser("~"),'/UIUC/fa25/CS441/final_projecf/repo/web_app/models' + MODEL_FILE)
TARGET_STATS = ['Pitches', 'SO_LHB', 'SO_RHB', 'IP_Outs', 'ER', 'WHIP'] 
FULL_START_YEAR = 2018
FULL_END_YEAR = 2023 

PA_EVENTS = [
    'strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 
    'triple', 'home_run', 'field_out', 'force_out', 'grounded_into_double_play',
    'double_play', 'triple_play', 'fielders_choice', 'fielders_choice_out'
]


# Mapping of common team names to the 3-letter abbreviation used in Statcast data
TEAM_ABBREVIATIONS = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL', 
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW', 
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL', 
    'Detroit Tigers': 'DET', 'Miami Marlins': 'MIA', 'Houston Astros': 'HOU', 
    'Kansas City Royals': 'KCR', 'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 
    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM', 
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI', 
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG', 
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR', 
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
    # Common abbreviations/short names
    'D-backs': 'ARI', 'Braves': 'ATL', 'Orioles': 'BAL', 'Red Sox': 'BOS', 'Cubs': 'CHC', 
    'White Sox': 'CHW', 'Reds': 'CIN', 'Guardians': 'CLE', 'Rockies': 'COL', 'Tigers': 'DET', 
    'Marlins': 'MIA', 'Astros': 'HOU', 'Royals': 'KCR', 'Angels': 'LAA', 'Dodgers': 'LAD', 
    'Brewers': 'MIL', 'Twins': 'MIN', 'Mets': 'NYM', 'Yankees': 'NYY', 'Athletics': 'OAK', 
    'Phillies': 'PHI', 'Pirates': 'PIT', 'Padres': 'SDP', 'Giants': 'SFG', 'Mariners': 'SEA', 
    'Cardinals': 'STL', 'Rays': 'TBR', 'Rangers': 'TEX', 'Blue Jays': 'TOR', 'Nationals': 'WSN',
    'STL': 'STL', 'NYY': 'NYY', 'MIL': 'MIL'
}

def get_team_abbreviation(team_name):
    normalized_name = team_name.strip().title()
    for key, value in TEAM_ABBREVIATIONS.items():
        if normalized_name in key or normalized_name == value or team_name.upper() == value:
            return value
    return None

def get_pitcher_id(name):
    name_parts = name.split()
    if len(name_parts) < 2:
        print("Error: Please provide a full name (first and last).")
        return None, None
        
    last_name = name_parts[-1]
    first_name = name_parts[0]
    try:
        lookup_df = playerid_lookup(last=last_name, first=first_name)
        if not lookup_df.empty:
            pitcher_row = lookup_df.iloc[0]
            player_id = pitcher_row['key_mlbam']
            full_name = f"{pitcher_row['name_first']} {pitcher_row['name_last']}"
            return player_id, full_name
        else:
            print(f"Pitcher not found")
            return None, None
    except Exception as e:
        print(f"Error during player lookup: {e}")
        return None, None

# Get data in chunks or else we run out of memory
def process_and_append_chunk(start_dt, end_dt, data_file, full_process_history=None):
    df_raw = statcast(start_dt=start_dt, end_dt=end_dt) 
    
    df_raw['game_date'] = pd.to_datetime(df_raw['game_date'])
    if 'pitcher' in df_raw.columns:
            df_raw.rename(columns={'pitcher': 'pitcher_id'}, inplace=True) 
    

    df_master_chunk = aggregate_and_engineer_features(df_raw, full_process_history) 
    
    if not df_master_chunk.empty:
        header = not os.path.exists(data_file)
        mode = 'w' if header else 'a'
        # save chunk to csv
        df_master_chunk.to_csv(data_file, mode=mode, header=header, index=False)

    del df_raw
    del df_master_chunk
    gc.collect()

def aggregate_and_engineer_features(df_raw, history_df=None):
    df = df_raw.dropna(subset=['pitch_type']).copy()
    df_pa = df[df['events'].isin(PA_EVENTS)].copy()
    df_pa['stand_type'] = df_pa['stand'].apply(lambda x: '_LHB' if x == 'L' else '_RHB')

    # Aggregation and Merge
    split_agg = df_pa.groupby(['game_date', 'pitcher_id', 'stand_type']).agg(
        SO=('events', lambda x: (x == 'strikeout').sum()),
        H=('events', lambda x: x.isin(['single', 'double', 'triple', 'home_run']).sum()),
        BB=('events', lambda x: (x == 'walk').sum()),
        PA=('events', 'count')
    ).reset_index()

    split_agg_pivot = split_agg.pivot_table(
        index=['game_date', 'pitcher_id'], columns='stand_type', 
        values=['SO', 'H', 'BB', 'PA'], fill_value=0
    ).reset_index()
    split_agg_pivot.columns = [f'{stat}{stand}' for stat, stand in split_agg_pivot.columns]
    
    agg_totals = df.groupby(['game_date', 'pitcher_id']).agg(
        Pitches=('pitch_type', 'count'),
        Outs_Recorded=('outs_when_up', lambda x: x[x.diff().fillna(1) != 0].sum()),
        ER=('delta_run_exp', lambda x: x[x > 0].sum()),
        HomeTeam=('home_team', 'first'),
        AwayTeam=('away_team', 'first'),
        GameID=('game_pk', 'first')
    ).reset_index()

    # Merge Aggregations
    df_master = pd.merge(agg_totals, split_agg_pivot, on=['game_date', 'pitcher_id'], how='left')
    
    required_split_cols = [f'{stat}{stand}' for stat in ['SO', 'H', 'BB', 'PA'] for stand in ['_LHB', '_RHB']]
    for col in required_split_cols:
        if col not in df_master.columns:
            df_master[col] = 0
    
    df_master.fillna(0, inplace=True)

    # Metric Calculation
    df_master['Year'] = df_master['game_date'].dt.year 
    df_master['IP_Outs'] = df_master['Outs_Recorded'] / 3.0
    df_master['Total_H'] = df_master['H_LHB'] + df_master['H_RHB']
    df_master['Total_BB'] = df_master['BB_LHB'] + df_master['BB_RHB']
    df_master['Total_PA'] = df_master['PA_LHB'] + df_master['PA_RHB']
    df_master['WHIP'] = (df_master['Total_BB'] + df_master['Total_H']) / df_master['IP_Outs']
    df_master.replace([np.inf, -np.inf], np.nan, inplace=True) 
    
    # Rolling Features
    if history_df is not None and not history_df.empty:
        # Check for necessary columns for rolling calculations
        required_cols = ['game_date', 'pitcher_id', 'Total_H', 'Total_PA']
        if not all(col in history_df.columns for col in required_cols):
             print("Historical data load error")
             df_combined = df_master.copy() 
        else:
            history_df_subset = history_df[required_cols]
            df_combined = pd.concat([history_df_subset, df_master], ignore_index=True)
    else:
        df_combined = df_master.copy()

    df_combined = df_combined.sort_values(by=['pitcher_id', 'game_date']).reset_index(drop=True)

    pitcher_groups = df_combined.groupby('pitcher_id')
   
    df_combined['Past_H_Sum'] = pitcher_groups['Total_H'].cumsum() - df_combined['Total_H']
    df_combined['Past_PA_Sum'] = pitcher_groups['Total_PA'].cumsum() - df_combined['Total_PA']


    df_combined['Past_AVG'] = df_combined['Past_H_Sum'] / df_combined['Past_PA_Sum']
    df_chunk_result = df_combined[df_combined['game_date'].isin(df_master['game_date'])].copy()
    
    # Replace division by zero/inf/NaN for first games with overall average
    mean_past_avg = df_chunk_result['Past_AVG'].replace([np.inf, -np.inf], np.nan).mean()
    df_chunk_result['Past_AVG'] = df_chunk_result['Past_AVG'].replace([np.inf, -np.inf], np.nan).fillna(mean_past_avg)

    # Determine Opponent and Home/Away Status
    df_chunk_result['OppTeam'] = np.where(df_chunk_result['HomeTeam'] == df_chunk_result['AwayTeam'], df_chunk_result['AwayTeam'], df_chunk_result['AwayTeam']) 
    df_chunk_result['HomeAway'] = np.where(df_chunk_result['HomeTeam'] == df_chunk_result['AwayTeam'], 'Neutral', 
                                          np.where(df_chunk_result['HomeTeam'].str.contains(r'[A-Z]{3}'), 'Home', 'Away'))
    
    final_cols = ['game_date', 'pitcher_id', 'Year', 'OppTeam', 'HomeAway', 'Past_AVG',
                  'SO_LHB', 'SO_RHB', 'H_LHB', 'H_RHB', 'BB_LHB', 'BB_RHB', 
                  'IP_Outs', 'ER', 'WHIP', 'Pitches', 'Total_PA']

    # Filter to only final columns that actually exist in the dataframe
    df_chunk_result = df_chunk_result[list(set(final_cols) & set(df_chunk_result.columns))]
    df_chunk_result = df_chunk_result.dropna(subset=['WHIP', 'Past_AVG']) 
    return df_chunk_result


def prepare_and_split_data(df):
    for col in TARGET_STATS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=TARGET_STATS)

    df = df[df['Pitches'] >= 65]
    df = df[df['IP_Outs'] > 0] 
    
    # prevent log(0)
    epsilon = 1e-6
    df['WHIP_LOG'] = np.log2(df['WHIP'] + epsilon) 
    
    # Ensure all inputs are non-negative
    df = df[df['Pitches'] >= 0].copy()
    df = df[df['SO_LHB'] >= 0].copy()
    df = df[df['SO_RHB'] >= 0].copy()
    df = df[df['ER'] >= 0].copy()
    df = df[df['IP_Outs'] <= 9.0].copy()


    df['SO_LHB_log'] = np.sqrt(df['SO_LHB'])
    df['SO_RHB_log'] = np.sqrt(df['SO_RHB'])
    df['ER_log'] = np.sqrt(df['ER'])
    df['Pitches_Log'] = np.sqrt(df['Pitches'])

    # drop the vars we are predicting
    X = df.drop(columns=['game_date', 'pitcher_id'] + TARGET_STATS, errors='ignore')
    y = df[TARGET_STATS]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna()
    y = y.loc[X.index]

    X = pd.get_dummies(X, drop_first=True)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train_val, y_train_val, X_test, y_test

def train_and_evaluate_models(X_train_val, y_train_val):
    models = {
        # 'Adam Neural Net (MLP)': MLPRegressor(
        #     solver='adam',
        #     max_iter=5000, 
        #     early_stopping=True, 
        #     n_iter_no_change=20, 
        #     validation_fraction=0.15,
        #     learning_rate='adaptive')
        'linear' : LinearRegression()
        # 'randomForest' : RandomForestRegressor(n_estimators=200)
    }
    
    results = {}

    for target_name in TARGET_STATS:
        y_target = y_train_val[target_name]
        results[target_name] = {}
        
        kf = KFold(n_splits=10, shuffle=True)
        
        for name, model in models.items():
            mse_scores = -cross_val_score(
                model, X_train_val, y_target, 
                cv=kf, scoring='neg_mean_squared_error', n_jobs=1
            )
            avg_rmse = np.sqrt(mse_scores).mean()
            results[target_name][name] = avg_rmse

            
    return results, models


def load_or_train_model(X_tv, y_tv, model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            rf_model_data = pickle.load(f)
            rf_model = rf_model_data['model']
            cv_results = rf_model_data['cv_results']
            print("Model loaded successfully. Skipping retraining.")
            return rf_model, cv_results
            
    
    cv_results, trained_models = train_and_evaluate_models(X_tv, y_tv)
    
    rf_model = LinearRegression()
    # rf_model = RandomForestRegressor(n_estimators=200)
    # rf_model = MLPRegressor(
    #     solver='adam',
    #     max_iter=5000,
    #     early_stopping=True,
    #     n_iter_no_change=20,
    #     validation_fraction=0.15,
    #     learning_rate='adaptive'
    # )
    rf_model.fit(X_tv, y_tv)
    
    rf_model_data = {
        'model': rf_model,
        'cv_results': cv_results,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model_data, f)

    return rf_model, cv_results

def create_performance_visuals(cv_results, y_test, y_pred_test, target_stats, best_model_name="MLP Regressor"):
    df_cv = pd.DataFrame(cv_results).T
    df_cv.index.name = 'Target'
    key_targets = TARGET_STATS
    
    fig, axes = plt.subplots(len(key_targets), 2, figsize=(16, 6 * len(key_targets)))
    if len(key_targets) == 1:
        axes = axes.reshape(1, 2)
        
    for i, target in enumerate(key_targets):
        y_actual = y_test[target]
        y_predicted = y_pred_test[:, target_stats.index(target)]
        residuals = y_actual - y_predicted
        
        # Actual vs. Predicted Scatter Plot
        ax1 = axes[i, 0]
        sns.scatterplot(x=y_actual, y=y_predicted, ax=ax1, color='#1f77b4', alpha=0.6)
        
        # Add a line of perfect prediction (y=x)
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
        
        ax1.set_title(f'{target} - Actual vs. Predicted ({best_model_name})', fontsize=14)
        ax1.set_xlabel('Actual Value', fontsize=12)
        ax1.set_ylabel('Predicted Value', fontsize=12)
        ax1.legend()
        
        # Residual Plot
        ax2 = axes[i, 1]
        sns.scatterplot(x=y_predicted, y=residuals, ax=ax2, color='#ff7f0e', alpha=0.6)
        ax2.hlines(y=0, xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1], colors='k', linestyles='--', alpha=0.7)
        
        ax2.set_title(f'{target} - Residuals Plot ({best_model_name})', fontsize=14)
        ax2.set_xlabel('Predicted Value', fontsize=12)
        ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        
    plt.tight_layout()
    plt.savefig('output/actual_vs_predicted_and_residuals.png')
    plt.close()

def predict_new_game(model, feature_columns, year, opponent_team, home_away, past_avg, historical_log_stats):
    all_features = {
        # 'Year': year,
        'Past_AVG': past_avg,
        'OppTeam': opponent_team,
        'HomeAway': home_away,
        **historical_log_stats
    }
    
    new_data = pd.DataFrame([all_features])
    
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)
    new_features = new_data_encoded.reindex(columns=feature_columns, fill_value=0)
    new_features = new_features.filter(items=feature_columns)

    prediction = model.predict(new_features)
    
    results_df = pd.DataFrame(prediction.reshape(1, -1), columns=TARGET_STATS)
    results_df = results_df.clip(lower=0)
    
    return results_df

if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        all_years = list(range(FULL_START_YEAR, FULL_END_YEAR + 1))
        history_df = pd.DataFrame()
        
        for i in range(0, len(all_years), 2):
            chunk_years = all_years[i:i + 2]
            start_year = chunk_years[0]
            end_year = chunk_years[-1]
            
            start_date = f'{start_year}-03-20' 
            end_date = f'{end_year}-10-01' 

            process_and_append_chunk(start_date, end_date, DATA_FILE, history_df)
            
            if os.path.exists(DATA_FILE):
                history_df = pd.read_csv(DATA_FILE, parse_dates=['game_date'])
            else:
                history_df = pd.DataFrame()

    if os.path.exists(DATA_FILE):
        df_full = pd.read_csv(DATA_FILE, parse_dates=['game_date'])
    else:
        df_full = pd.DataFrame()
        
    gc.collect() 

    if df_full.empty or len(df_full) < 10: 
        print("Could not proceed due to insufficient data.")

        # apply transformations
        def apply_transformations(df):
            df_temp = df.copy()
            for col in TARGET_STATS:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
            df_temp = df_temp.dropna(subset=TARGET_STATS)
            df_temp = df_temp[df_temp['Pitches'] >= 65]
            df_temp = df_temp[df_temp['IP_Outs'] > 0].copy()

            # avoid log(0)
            epsilon = 1e-6
            df_temp['WHIP_LOG'] = np.log2(df_temp['WHIP'] + epsilon) 
            df_temp['SO_LHB_log'] = np.sqrt(df_temp['SO_LHB'])
            df_temp['SO_RHB_log'] = np.sqrt(df_temp['SO_RHB'])
            df_temp['ER_log'] = np.sqrt(df_temp['ER'])
            df_temp['Pitches_Log'] = np.sqrt(df_temp['Pitches'])
            
            return df_temp

        df_full = apply_transformations(df_full)
        
        X_tv, y_tv, X_test, y_test = prepare_and_split_data(df_full)
        rf_model, cv_results = load_or_train_model(X_tv, y_tv, MODEL_FILE)
        if rf_model is not None:
            y_pred_test = rf_model.predict(X_test)
            r2 = r2_score(y_test, y_pred_test)
            
            create_performance_visuals(cv_results, y_test, y_pred_test, TARGET_STATS, best_model_name="MLP Regressor")

            pitcher_name = input("Enter Pitcher's Full Name (e.g., Zack Greinke): ")
            opponent_team_name = input("Enter Opponent Team Name (e.g., Cardinals or STL): ")
            home_or_away = str(input("Is this pitcher on the home or away team? (Home/Away): "))
            
            pitcher_id, found_name = get_pitcher_id(pitcher_name)
            opponent_team_abbr = get_team_abbreviation(opponent_team_name)

            if pitcher_id and opponent_team_abbr:

                pitcher_history = df_full[df_full['pitcher_id'] == pitcher_id].sort_values('game_date')
                historical_log_stats = {}
                log_cols = ['WHIP_LOG', 'SO_LHB_log', 'SO_RHB_log', 'ER_log', 'Pitches_Log']

                if not pitcher_history.empty:
                    pitcher_games = pitcher_history.copy()
                    pitcher_vs_team = pitcher_games[pitcher_games['OppTeam'] == opponent_team_abbr]
                    numeric_cols = df_full.select_dtypes(include=['number']).columns
                    excluded_cols = set(TARGET_STATS + ['pitcher_id', 'game_date'])

                    historical_log_stats = {}

                    if not pitcher_vs_team.empty:
                        for col in numeric_cols:
                            if col not in excluded_cols:
                                historical_log_stats[col] = pitcher_vs_team[col].mean()

                        example_past_avg = pitcher_vs_team['Past_AVG'].mean()

                    else:
                        for col in numeric_cols:
                            if col not in excluded_cols:
                                historical_log_stats[col] = pitcher_games[col].mean()

                        example_past_avg = pitcher_games['Past_AVG'].mean()
                    
                else:  
                    for col in log_cols:
                        historical_log_stats[col] = df_full[col].mean()
                        
                    example_past_avg = df_full['Past_AVG'].mean()

                new_game_prediction = predict_new_game(
                    model=rf_model,
                    feature_columns=X_tv.columns,
                    year=FULL_END_YEAR,
                    opponent_team=opponent_team_abbr, 
                    home_away=home_or_away,
                    past_avg=example_past_avg,
                    historical_log_stats=historical_log_stats
                )
                new_game_prediction = new_game_prediction.clip(lower=0)
                print(new_game_prediction.round(2).to_markdown(index=False))
                print("\n")