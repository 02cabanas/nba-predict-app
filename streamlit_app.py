import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import subprocess


@st.cache_data
def load_data():
    file_path = 'line_score.csv'
    game_data = pd.read_csv(file_path)
    game_data['game_date_est'] = pd.to_datetime(game_data['game_date_est'])
    game_data['season'] = game_data['game_date_est'].apply(
        lambda x: f"{x.year}-{x.year + 1}" if x.month >= 10 else f"{x.year - 1}-{x.year}"
    )
    return game_data

game_data = load_data()

@st.cache_data
def compute_team_stats(df):
    stats = []
    team_stats = {}
    for _, row in df.sort_values('game_date_est').iterrows():
        date = row['game_date_est']
        home_team = row['team_id_home']
        away_team = row['team_id_away']
        home_points = row['pts_home']
        away_points = row['pts_away']

        if date.month >= 10:
            current_season = f"{date.year}-{date.year + 1}"
        else:
            current_season = f"{date.year - 1}-{date.year}"

        for team, points_for, points_against, is_home in [
            (home_team, home_points, away_points, True),
            (away_team, away_points, home_points, False),
        ]:
            if team not in team_stats:
                team_stats[team] = {
                    'total_points': 0,
                    'total_points_against': 0,
                    'home_wins': 0,
                    'home_losses': 0,
                    'away_wins': 0,
                    'away_losses': 0,
                }

            stats_dict = team_stats[team]
            stats_dict['total_points'] += points_for
            stats_dict['total_points_against'] += points_against

            if is_home:
                if points_for > points_against:
                    stats_dict['home_wins'] += 1
                else:
                    stats_dict['home_losses'] += 1
            else:
                if points_for > points_against:
                    stats_dict['away_wins'] += 1
                else:
                    stats_dict['away_losses'] += 1

            stats.append({
                'date': date,
                'team': team,
                'season': current_season,
                'total_points': stats_dict['total_points'],
                'total_points_against': stats_dict['total_points_against'],
                'home_wins': stats_dict['home_wins'],
                'home_losses': stats_dict['home_losses'],
                'away_wins': stats_dict['away_wins'],
                'away_losses': stats_dict['away_losses'],
            })
    return pd.DataFrame(stats).drop_duplicates()

team_stats = compute_team_stats(game_data)

@st.cache_data
def train_model():
    features = [
        'home_total_points', 'home_total_points_against', 'home_home_wins',
        'home_home_losses', 'away_total_points', 'away_total_points_against',
        'away_home_wins', 'away_home_losses'
    ]

    processed_data = game_data.merge(
        team_stats.rename(columns={
            'team': 'team_id_home',
            'total_points': 'home_total_points',
            'total_points_against': 'home_total_points_against',
            'home_wins': 'home_home_wins',
            'home_losses': 'home_home_losses',
            'away_wins': 'home_away_wins',
            'away_losses': 'home_away_losses'
        }),
        left_on=['game_date_est', 'team_id_home'],
        right_on=['date', 'team_id_home'],
        how='left'
    ).merge(
        team_stats.rename(columns={
            'team': 'team_id_away',
            'total_points': 'away_total_points',
            'total_points_against': 'away_total_points_against',
            'home_wins': 'away_home_wins',
            'home_losses': 'away_home_losses',
            'away_wins': 'away_away_wins',
            'away_losses': 'away_away_losses'
        }),
        left_on=['game_date_est', 'team_id_away'],
        right_on=['date', 'team_id_away'],
        how='left'
    )

    processed_data = processed_data.drop(columns=['date_x', 'date_y'])
    processed_data['home_win'] = (processed_data['pts_home'] > processed_data['pts_away']).astype(int)

    X = processed_data[features]
    y = processed_data['home_win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, features

model, features = train_model()

def predict_game(team_abbreviation_home, team_abbreviation_away, game_date):
    home_team_data = game_data[game_data['team_abbreviation_home'] == team_abbreviation_home]
    away_team_data = game_data[game_data['team_abbreviation_away'] == team_abbreviation_away]

    if home_team_data.empty:
        st.error(f"No data available for the home team: {team_abbreviation_home}")
        return
    if away_team_data.empty:
        st.error(f"No data available for the away team: {team_abbreviation_away}")
        return

    home_team_id = home_team_data['team_id_home'].iloc[0]
    away_team_id = away_team_data['team_id_away'].iloc[0]

    season = f"{game_date.year}-{game_date.year + 1}" if game_date.month >= 10 else f"{game_date.year - 1}-{game_date.year}"
    season_games = game_data[
        (game_data['season'] == season) &
        (game_data['game_date_est'] < pd.Timestamp(game_date)) &
        (
            (game_data['team_id_home'] == home_team_id) |
            (game_data['team_id_away'] == home_team_id) |
            (game_data['team_id_home'] == away_team_id) |
            (game_data['team_id_away'] == away_team_id)
        )
    ]

    if season_games.empty:
        st.error(f"No data available for the selected teams in the season {season}.")
        return

    # Calculate cumulative stats for the home and away teams
    home_total_points = season_games[season_games['team_id_home'] == home_team_id]['pts_home'].sum() + \
                        season_games[season_games['team_id_away'] == home_team_id]['pts_away'].sum()

    home_total_points_against = season_games[season_games['team_id_home'] == home_team_id]['pts_away'].sum() + \
                                season_games[season_games['team_id_away'] == home_team_id]['pts_home'].sum()

    home_home_wins = len(season_games[season_games['team_id_home'] == home_team_id])
    home_home_losses = len(season_games[season_games['team_id_away'] == home_team_id])

    away_total_points = season_games[season_games['team_id_home'] == away_team_id]['pts_home'].sum() + \
                        season_games[season_games['team_id_away'] == away_team_id]['pts_away'].sum()

    away_total_points_against = season_games[season_games['team_id_home'] == away_team_id]['pts_away'].sum() + \
                                season_games[season_games['team_id_away'] == away_team_id]['pts_home'].sum()

    away_home_wins = len(season_games[season_games['team_id_home'] == away_team_id])
    away_home_losses = len(season_games[season_games['team_id_away'] == away_team_id])

   
    input_data = pd.DataFrame([{
        'home_total_points': home_total_points,
        'home_total_points_against': home_total_points_against,
        'home_home_wins': home_home_wins,
        'home_home_losses': home_home_losses,
        'away_total_points': away_total_points,
        'away_total_points_against': away_total_points_against,
        'away_home_wins': away_home_wins,
        'away_home_losses': away_home_losses,
    }])

   
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0  

    # Make the prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]  # Get probabilities for both classes

    
    result = f"{team_abbreviation_home} Wins" if prediction == 1 else f"{team_abbreviation_away} Wins"
    winning_team_probability = probabilities[1] if prediction == 1 else probabilities[0]

    
    st.markdown(
        f"""
        <div style="text-align: left; font-size: 30px; color: white; font-weight: bold;">
            Prediction: <span style="color: green;">{result}</span>
        </div>
        <div style="text-align: left; font-size: 20px; color: white;">
            Probability: {winning_team_probability:.2%} chance of winning
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display bar charts for points and wins/losses
    st.subheader("Points Scored and Points Against")
    points_data = pd.DataFrame({
        'Feature': [
            f"{team_abbreviation_home} Total Points",
            f"{team_abbreviation_home} Points Against",
            f"{team_abbreviation_away} Total Points",
            f"{team_abbreviation_away} Points Against",
        ],
        'Value': [
            home_total_points,
            home_total_points_against,
            away_total_points,
            away_total_points_against,
        ],
    })
    st.bar_chart(points_data.set_index('Feature'))

    st.subheader("Wins and Losses")
    wins_losses_data = pd.DataFrame({
        'Feature': [
            f"{team_abbreviation_home} Home Wins",
            f"{team_abbreviation_home} Home Losses",
            f"{team_abbreviation_away} Away Wins",
            f"{team_abbreviation_away} Away Losses",
        ],
        'Value': [
            home_home_wins,
            home_home_losses,
            away_home_wins,
            away_home_losses,
        ],
    })
    st.bar_chart(wins_losses_data.set_index('Feature'))





# Sidebar navigation
st.sidebar.title("Menu") 
home_button = st.sidebar.button("Home")
prediction_button = st.sidebar.button("Win/Loss Predictor")
data_insights_button = st.sidebar.button("Data Insights")
about_button = st.sidebar.button("About")
over_under_button = st.sidebar.button("Over/Under Predictor")
over_under_visuals_button = st.sidebar.button("Over/Under Visuals")

# Set default page if no button is pressed
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Update page state based on button click
if home_button:
    st.session_state.page = "Home"
elif prediction_button:
    st.session_state.page = "Prediction"
elif data_insights_button:
    st.session_state.page = "Data Insights"
elif about_button:
    st.session_state.page = "About"
elif over_under_button:
    st.session_state.page = "Over/Under Predictor"
elif over_under_visuals_button:
    st.session_state.page = "Over/Under Visuals"

# Render the selected page
if st.session_state.page == "Home":
    st.title("NBA Predictor App")
    st.write("Welcome to the NBA Predictor App.")
    st.subheader("Intro to Data Science Final Project")
    st.markdown("""
**CAP 5768**<br>
Dr. Juhàsz<br>
Mon 5pm - 7:40pm
""", unsafe_allow_html=True)
    st.markdown("""
**Team Members:**<br>
Ernesto Gomila<br>
Brandon Rodriguez<br>
Alfredo<br>
Abel<br>
""", unsafe_allow_html=True)
    


elif st.session_state.page == "Prediction":
    st.title("Make a Prediction")
    with st.form("prediction_form"):
        team_abbreviation_home = st.text_input("Enter Home Team Abbreviation:")
        team_abbreviation_away = st.text_input("Enter Away Team Abbreviation:")
        game_date = st.date_input("Select Game Date:")
        submit = st.form_submit_button("Predict")

    if submit:
        predict_game(team_abbreviation_home, team_abbreviation_away, game_date)

elif st.session_state.page == "Data Insights":
    st.title("Data Insights")
    st.write(game_data.head())

elif st.session_state.page == "About":
    st.title("About")
    st.write("This app predicts NBA game outcomes.")



elif st.session_state.page == "Over/Under Predictor":
    st.title("Over/Under Predictor")

    with st.form("over_under_form"):
        player = st.text_input("Enter Player Name:")
        team1 = st.text_input("Enter Team 1 Abbreviation (Player's Team):")
        team2 = st.text_input("Enter Team 2 Abbreviation (Opponent):")
        game_date = st.date_input("Enter Game Date:")
        points = st.number_input("Enter Points Threshold:", min_value=0, step=1, value=20)
        submit = st.form_submit_button("Predict Over/Under")

    if submit:
        if not all([player, team1, team2, game_date, points]):
            st.error("Please fill in all the fields.")
        else:
            # Construct the command to call your model
            command = f"python3 over_under.py predict --player \"{player}\" --team1 \"{team1}\" --team2 \"{team2}\" --date \"{game_date}\" --points {int(points)}"
            
            # Use subprocess to run the command
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Prediction Complete!")
                    st.text(result.stdout)
                else:
                    st.error("Error in prediction.")
                    st.text(result.stderr)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.subheader("Team Abbreviations")
    st.markdown("""
    Here are common NBA team abbreviations:
    - **Atlanta Hawks** → ATL
    - **Boston Celtics** → BOS
    - **Brooklyn Nets** → BKN
    - **Charlotte Hornets** → CHA
    - **Chicago Bulls** → CHI
    - **Cleveland Cavaliers** → CLE
    - **Dallas Mavericks** → DAL
    - **Denver Nuggets** → DEN
    - **Detroit Pistons** → DET
    - **Golden State Warriors** → GSW
    - **Houston Rockets** → HOU
    - **Indiana Pacers** → IND
    - **LA Clippers** → LAC
    - **Los Angeles Lakers** → LAL
    - **Memphis Grizzlies** → MEM
    - **Miami Heat** → MIA
    - **Milwaukee Bucks** → MIL
    - **Minnesota Timberwolves** → MIN
    - **New Orleans Pelicans** → NOP
    - **New York Knicks** → NYK
    - **Oklahoma City Thunder** → OKC
    - **Orlando Magic** → ORL
    - **Philadelphia 76ers** → PHI
    - **Phoenix Suns** → PHX
    - **Portland Trail Blazers** → POR
    - **Sacramento Kings** → SAC
    - **San Antonio Spurs** → SAS
    - **Toronto Raptors** → TOR
    - **Utah Jazz** → UTA
    - **Washington Wizards** → WAS
    """)

elif st.session_state.page == "Over/Under Visuals":
    st.title("Over/Under Visuals")
    st.markdown(
        "These visuals show the accuracy and feature importances for predicting over/under on sample thresholds for player points. "
        "Each section corresponds to predictions for a specific point threshold (10, 20, and 30)."
    )

    # 10-Point Section
    st.subheader("10-Point Threshold")
    st.image("over_under_visuals/10_kpi.png", caption="Model KPI's for 10-Point Threshold")
    st.image("over_under_visuals/10_accuracy.png", caption="Accuracy for 10-Point Threshold")
    st.markdown("This chart shows the prediction outcomes (correct and incorrect) for a 10-point threshold on the test set.")
    st.image("over_under_visuals/10_feature_imp.png", caption="Feature Importances for 10-Point Threshold")
    st.markdown("This chart shows the most important features contributing to predictions for a 10-point threshold."
                "In this case we see that for low points the team the player is on is more important for wether they will be over/under the low 10-point threshold.")

    # 20-Point Section
    st.subheader("20-Point Threshold")
    st.image("over_under_visuals/20_kpi.png", caption="Model KPI's for 20-Point Threshold")
    st.image("over_under_visuals/20_accuracy.png", caption="Accuracy for 20-Point Threshold")
    st.markdown("This chart shows the prediction outcomes (correct and incorrect) for a 20-point threshold on the test set.")
    st.image("over_under_visuals/20_feature_imp.png", caption="Feature Importances for 20-Point Threshold")
    st.markdown("This chart shows the most important features contributing to predictions for a 20-point threshold."
                "In this case we see that for medium points the the player; not the team, is on is more important for wether they will be over/under the low 10-point threshold."
                "This also gives us insight into the player the model is most confident/decisive in its prediction since the weights are high for these player names.")

    # 30-Point Section
    st.subheader("30-Point Threshold")
    st.image("over_under_visuals/30_kpi.png", caption="Model KPI's for 30-Point Threshold")
    st.image("over_under_visuals/30_accuracy.png", caption="Accuracy for 30-Point Threshold")
    st.markdown("This chart shows the prediction outcomes (correct and incorrect) for a 30-point threshold on the test set."
                "We see that the higher the point threshold it seems the more accurate the model is in its prediction.")
    st.image("over_under_visuals/30_feature_imp.png", caption="Feature Importances for 30-Point Threshold")
    st.markdown("This chart shows the most important features contributing to predictions for a 30-point threshold."
                "In this case we also see that for medium points the the player; not the team, is on is more important for wether they will be over/under the low 10-point threshold."
                "This also gives us insight into the player the model is most confident/decisive in its prediction since the weights are high for these player names.")