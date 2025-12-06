from flask import Flask, render_template, request
import pickle
from main_script import load_data
import requests
from bs4 import BeautifulSoup
from pybaseball import playerid_lookup
from main_script import (
    FEATURE_COLUMNS,
    TARGET_STATS,
    get_pitcher_id,
    get_team_abbreviation,
    predict_new_game,
    FULL_END_YEAR,
)
import os

app = Flask(__name__)
df_full = load_data()

# Model paths
MODEL_PATHS = {
    "adamNet": os.path.join(os.path.expanduser("~"),"UIUC/fa25/CS441/final_projecf/repo/web_app/models/adamNet.pkl"),
    "linearRegression": os.path.join(os.path.expanduser("~"),"UIUC/fa25/CS441/final_projecf/repo/web_app/models/linearRegression.pkl"),
    "randomForest": os.path.join(os.path.expanduser("~"),"UIUC/fa25/CS441/final_projecf/repo/web_app/models/randomForest.pkl")
}

def load_model_file(model_name):
    with open(MODEL_PATHS[model_name], "rb") as f:
        return pickle.load(f)["model"]

def get_pitcher_photo_url(name):
    first, last = name.split()
    data = playerid_lookup(last, first)
    player_id = data.key_bbref.iloc[0]
    first_letter = str(player_id)[0]
    url = f"https://www.baseball-reference.com/players/{first_letter}/{player_id}.shtml"

    response = requests.get(url)
    if response.status_code == 200:
        page_content = response.text

    soup = BeautifulSoup(page_content, 'html.parser')

    media_div = soup.find('div', class_='media-item multiple')

    if media_div:
        img_tags = media_div.find_all('img')

        headshot_urls = [img['src'] for img in img_tags]
        for url in headshot_urls:
            return(url)

def get_team_logo_url(team_abbr):
    return f' https://cdn.ssref.net/req/202510241/tlogo/br/{team_abbr}.png'

# This index function was written by Gemini
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    pitcher_img = None
    team_logo_img = None
    pitcher_display = None
    model_name = None
    opponent_team_name = None

    if request.method == "POST":
        model_name = request.form.get("model")
        pitcher_name = request.form.get("pitcher")
        opponent_team_name = request.form.get("team")
        home_or_away = request.form.get("homeAway")

        try:
            model = load_model_file(model_name)
        except Exception as e:
            error = str(e)
            return render_template("index.html", prediction=None, error=error)

        # Get IDs and abbreviations
        pitcher_id, pitcher_display = get_pitcher_id(pitcher_name)
        team_abbr = get_team_abbreviation(opponent_team_name)

        if not pitcher_id or not team_abbr:
            error = "Invalid pitcher name or opponent team."
            return render_template("index.html", prediction=None, error=error)

        # Build pitcher historical stats
        pitcher_history = df_full[df_full["pitcher_id"] == pitcher_id].sort_values("game_date")
        historical_log_stats = {}
        if not pitcher_history.empty:
            pitcher_games = pitcher_history.copy()

            # Filter only games against this opponent
            pitcher_vs_team = pitcher_games[pitcher_games['OppTeam'] == team_abbr]

            # Numeric columns only (safe for mean)
            numeric_cols = df_full.select_dtypes(include=['number']).columns

            # Columns we never average
            excluded_cols = set(TARGET_STATS + ['pitcher_id', 'game_date'])

        # Pitcher has history with other team
        if not pitcher_vs_team.empty:
            for col in numeric_cols:
                if col not in excluded_cols:
                    historical_log_stats[col] = pitcher_vs_team[col].mean()

            example_past_avg = pitcher_vs_team['Past_AVG'].mean()

        #Pitcher does not have history with other team
        else:
            for col in numeric_cols:
                if col not in excluded_cols:
                    historical_log_stats[col] = pitcher_games[col].mean()

            example_past_avg = pitcher_games['Past_AVG'].mean()

        pred = predict_new_game(
            model=model,
            feature_columns=FEATURE_COLUMNS,
            year=FULL_END_YEAR,
            opponent_team=team_abbr,
            home_away=home_or_away,
            past_avg=example_past_avg,
            historical_log_stats=historical_log_stats,
        )

        prediction = pred.to_dict(orient="records")[0]

        # Get images
        pitcher_img = get_pitcher_photo_url(pitcher_name)
        team_logo_img = get_team_logo_url(team_abbr)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        pitcher=pitcher_display,
        pitcher_img=pitcher_img,
        team_logo_img=team_logo_img,
        model_name=model_name,
        opponent_team_name=opponent_team_name
    )

if __name__ == "__main__":
    app.run(debug=True)
