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


# Configure plotting aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans']

# --- CONFIGURATION ---
DATA_FILE = os.path.join(os.path.expanduser("~"),'UIUC/fa25/CS441/final_projecf/repo/data/pitching_master_data_aggregated.csv')
TARGET_STATS = ['Pitches', 'SO_LHB', 'SO_RHB', 'IP_Outs', 'ER', 'WHIP'] 
FULL_START_YEAR = 2018
FULL_END_YEAR = 2023 

PA_EVENTS = [
    'strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 
    'triple', 'home_run', 'field_out', 'force_out', 'grounded_into_double_play',
    'double_play', 'triple_play', 'fielders_choice', 'fielders_choice_out'
]

# Mapping of common team names to the 3-letter abbreviation used in Statcast data
# ChatGPT generated these names
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

FEATURE_COLUMNS = [
    'BB_RHB', 'Total_PA', 'H_RHB', 'H_LHB', 'Past_AVG', 'Year', 'BB_LHB',
    'WHIP_LOG', 'SO_LHB_log', 'SO_RHB_log', 'ER_log', 'Pitches_Log',
    'HomeAway_Home', 'OppTeam_AZ', 'OppTeam_BAL', 'OppTeam_BOS',
    'OppTeam_CHC', 'OppTeam_CIN', 'OppTeam_CLE', 'OppTeam_COL',
    'OppTeam_CWS', 'OppTeam_DET', 'OppTeam_HOU', 'OppTeam_KC',
    'OppTeam_LAA', 'OppTeam_LAD', 'OppTeam_MIA', 'OppTeam_MIL',
    'OppTeam_MIN', 'OppTeam_NYM', 'OppTeam_NYY', 'OppTeam_OAK',
    'OppTeam_PHI', 'OppTeam_PIT', 'OppTeam_SD', 'OppTeam_SEA',
    'OppTeam_SF', 'OppTeam_STL', 'OppTeam_TB', 'OppTeam_TEX',
    'OppTeam_TOR', 'OppTeam_WSH'
]


def get_team_abbreviation(team_name):
    normalized_name = team_name.strip().title()
    for key, value in TEAM_ABBREVIATIONS.items():
        if normalized_name in key or normalized_name == value or team_name.upper() == value:
            return value
    return None

def get_pitcher_id(name):
    name_parts = name.split()
    if len(name_parts) < 2:
        print("Please provide a full name (first and last).")
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

def predict_new_game(model, feature_columns, year, opponent_team, home_away, past_avg, historical_log_stats):
    all_features = {
        'Past_AVG': past_avg,
        'OppTeam': opponent_team,
        'HomeAway': home_away,
        **historical_log_stats 
    }
    
    new_data = pd.DataFrame([all_features])
    # used pandas docs
    # chatGPT idea to use get_dummies
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)
    new_features = new_data_encoded.reindex(columns=feature_columns, fill_value=0)
    
    new_features = new_features.filter(items=feature_columns)

    prediction = model.predict(new_features)
    results_df = pd.DataFrame(prediction.reshape(1, -1), columns=TARGET_STATS)
    results_df = results_df.clip(lower=0)
    
    return results_df

def load_or_train_model(model_path):
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                # used pickle docs
                rf_model_data = pickle.load(f)
                rf_model = rf_model_data['model']
                cv_results = rf_model_data['cv_results']
                return rf_model, cv_results
        except:
            return
        
# chatgpt helped with 
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=['game_date'])
    else:
        df = pd.DataFrame()

    epsilon = 1e-6
    if not df.empty:
        df['WHIP_LOG'] = np.log2(df['WHIP'] + epsilon) 
        df['SO_LHB_log'] = np.sqrt(df['SO_LHB'])
        df['SO_RHB_log'] = np.sqrt(df['SO_RHB'])
        df['ER_log'] = np.sqrt(df['ER'])
        df['Pitches_Log'] = np.sqrt(df['Pitches'])
    
    return df