import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
test_button = st.sidebar.button("fredo test")

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

# Render the selected page
if st.session_state.page == "Home":

    # Title with emoji for a vibrant look
    st.markdown("""
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
            <h1 style="color: white; font-family: 'Arial', sans-serif; margin-bottom: 0;">🏀 NBA Predictor App</h1>
            <strong style="font-size: 20px; color: white; margin-top: 5px;">Elevate your gaming bets with our predictions, powered by data science!</strong>
        </div>
    """, unsafe_allow_html=True)

    # Welcome message with improved layout
    st.markdown("""
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52); padding: 15px; border-radius: 15px; text-align: center; margin-top: 20px; border: 1px solid #ddd; box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="color: white; font-family: 'Verdana', sans-serif;">Welcome to the NBA Predictor App</h2>
            <strong style="font-size: 18px; color: white; font-family: 'Georgia', serif;">
                Your ultimate tool for predicting NBA outcomes, powered by data science!
            </strong>
        </div>
    """, unsafe_allow_html=True)

    # Subtitle with better spacing
    st.markdown("""
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52);padding: 15px; border-radius: 15px; text-align: center; border: 1px solid #ddd; box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);margin-top: 30px; text-align: left;">
            <h3 style="color: #333; font-family: 'Helvetica', sans-serif;">📚 Intro to Data Science Final Project</h3>
            <Strong style="font-size: 16px; color: #444; line-height: 1.6;">
                <b>CAP 5768</b><br>
                <b>Instructor:</b> Dr. Juhàsz<br>
                <b>Schedule: Monday 5:00 PM - 7:40 PM</b> 
            </Strong>
        </div>
    """, unsafe_allow_html=True)

    # Team members with hover effect using CSS
    st.markdown("""
        <style>
            .team-member-list li {
                margin-bottom: 8px;
                font-size: 16px;
                color: #444;
                transition: color 0.3s ease;
            }
            .team-member-list li:hover {
                color: #0077b6;
            }
        </style>
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52);padding: 15px; border-radius: 15px; text-align: center; border: 1px solid #ddd; box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);margin-top: 30px; text-align: left;">
            <h3 style="color: #333; font-family: 'Helvetica', sans-serif;">🌟 Team Members:</h3>
            <ul class="team-member-list" style="list-style: none; padding: 0;">
                <li><Strong>Ernesto Gomila</Strong></li>
                <li><Strong>Brandon Rodriguez</Strong></li>
                <li><Strong>Alfredo Cal</Strong></li>
                <li><Strong>Abel</Strong></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style="margin-top: 40px; text-align: center; color: #aaa; font-size: 14px;">
        Made with 🧠 by the NBA Predictor Team | Powered by Streamlit
         </div>
    """, unsafe_allow_html=True)

    # Apply background color for the entire app
    st.markdown("""
        <style>
            body {
                background-color: #f4f4f4; /* Light off-white background for readability */
            }
            .main-content {
                padding: 20px;
            }
        </style>
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


# Cache data loading function
@st.cache_data
def load_data():
    line_score = pd.read_csv('line_score.csv')
    common_player_info = pd.read_csv('common_player_info.csv')
    draft_history = pd.read_csv('draft_history.csv')
    return line_score, common_player_info, draft_history


# Main app logic
if st.session_state.page == "Data Insights":
    st.title("Data Insights and Analysis")

    # Load data
    line_score, common_player_info, draft_history = load_data()

    # Preview datasets
    st.header("Dataset Previews")
    st.subheader("Line Score")
    st.dataframe(line_score.head())
    st.subheader("Common Player Info")
    st.dataframe(common_player_info.head())
    st.subheader("Draft History")
    st.dataframe(draft_history.head())

    # Standardize column names
    line_score.columns = line_score.columns.str.lower()
    common_player_info.columns = common_player_info.columns.str.lower()
    draft_history.columns = draft_history.columns.str.lower()

    # Analysis 1: Total Points Scored Per Team
    st.header("Analysis 1: Total Points Scored Per Team")
    if 'team_nickname_home' in line_score.columns and 'pts_home' in line_score.columns and \
            'team_nickname_away' in line_score.columns and 'pts_away' in line_score.columns:
        total_points_home = line_score.groupby('team_nickname_home')['pts_home'].sum().reset_index()
        total_points_home.columns = ['team', 'total_points']
        total_points_away = line_score.groupby('team_nickname_away')['pts_away'].sum().reset_index()
        total_points_away.columns = ['team', 'total_points']
        total_points = pd.concat([total_points_home, total_points_away]).groupby('team').sum().reset_index()

        fig = px.bar(
            total_points,
            x='team',
            y='total_points',
            title='Total Points Scored Per Team',
            labels={'total_points': 'Total Points'},
            color='total_points',
        )
        st.plotly_chart(fig)
    else:
        st.error("Relevant columns for total points analysis are missing in the dataset.")

    # Analysis 2: Top Teams by Wins
    st.header("Analysis 2: Top Teams by Wins")
    if 'team_wins_losses_home' in line_score.columns and 'team_nickname_home' in line_score.columns and \
            'team_wins_losses_away' in line_score.columns:
        # Clean and convert win/loss data
        line_score['wins_home'] = (
            line_score['team_wins_losses_home']
            .str.split('-')
            .str[0]
            .apply(lambda x: int(x) if x.isdigit() else None)
        )
        line_score['wins_away'] = (
            line_score['team_wins_losses_away']
            .str.split('-')
            .str[0]
            .apply(lambda x: int(x) if x.isdigit() else None)
        )

        # Aggregate wins
        total_wins = (
            line_score.groupby('team_nickname_home')['wins_home']
            .sum()
            .reset_index()
            .rename(columns={'team_nickname_home': 'team', 'wins_home': 'total_wins'})
        )
        fig = px.bar(
            total_wins,
            x='team',
            y='total_wins',
            title="Top Teams by Total Wins",
            labels={'total_wins': 'Wins'},
            color='total_wins',
        )
        st.plotly_chart(fig)
    else:
        st.error("Relevant columns to analyze wins are missing.")

    # Additional Insights
    st.header("Additional Insights")

    # Insight 3: Top 10 Players of All Time
    st.header("Insight 3: Top 10 Players of All Time")
    if 'player_id' in line_score.columns and 'pts_home' in line_score.columns and 'player_id' in common_player_info.columns:
        merged_data = line_score.merge(common_player_info, on='player_id', how='inner')

        top_players = (
            merged_data.groupby('player_name')['pts_home']
            .sum()
            .reset_index()
            .sort_values(by='pts_home', ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_players,
            x='player_name',
            y='pts_home',
            title="Top 10 Players of All Time by Total Points",
            labels={'pts_home': 'Total Points'},
            color='pts_home',
        )
        st.plotly_chart(fig)
    else:
        st.error("Columns required for Insight 3 are missing. Ensure `player_id` and `pts_home` are present.")

    # Insight 4: Most Consistent Players
    st.header("Insight 4: Top 10 Most Consistent Players (Avg. Points/Game)")
    if 'player_id' in line_score.columns and 'game_id' in line_score.columns and 'pts_home' in line_score.columns:
        merged_data = line_score.merge(common_player_info, on='player_id', how='inner')

        player_stats = (
            merged_data.groupby(['player_name', 'player_id'])
            .agg(total_points=('pts_home', 'sum'), total_games=('game_id', 'count'))
            .reset_index()
        )
        player_stats['avg_points_per_game'] = player_stats['total_points'] / player_stats['total_games']

        top_consistent_players = player_stats.sort_values(by='avg_points_per_game', ascending=False).head(10)

        fig = px.scatter(
            top_consistent_players,
            x='player_name',
            y='avg_points_per_game',
            size='total_points',
            title="Top 10 Most Consistent Players (Avg. Points/Game)",
            labels={'avg_points_per_game': 'Average Points/Game'},
            color='avg_points_per_game',
        )
        st.plotly_chart(fig)
    else:
        st.error("Columns required for Insight 4 are missing. Ensure `game_id` and `pts_home` are present.")

    # Insight 5: Top Teams in the Last Decade
    st.header("Insight 5: Top Teams in the Last Decade")
    if 'team_nickname_home' in line_score.columns and 'game_date' in line_score.columns and 'team_wins_losses_home' in line_score.columns:
        line_score['game_date'] = pd.to_datetime(line_score['game_date'], errors='coerce')
        recent_games = line_score[line_score['game_date'] >= pd.Timestamp.now() - pd.DateOffset(years=10)]

        recent_games['wins_home'] = (
            recent_games['team_wins_losses_home']
            .str.split('-')
            .str[0]
            .apply(lambda x: int(x) if x.isdigit() else None)
        )
        total_wins_recent = (
            recent_games.groupby('team_nickname_home')['wins_home']
            .sum()
            .reset_index()
            .rename(columns={'team_nickname_home': 'team', 'wins_home': 'total_wins'})
        )

        fig = px.bar(
            total_wins_recent.head(10),
            x='team',
            y='total_wins',
            title="Top Teams in the Last Decade",
            labels={'total_wins': 'Total Wins'},
            color='total_wins',
        )
        st.plotly_chart(fig)
    else:
        st.error("Columns required for Insight 5 are missing. Ensure `game_date` and `team_wins_losses_home` are present.")

elif st.session_state.page == "About":
    st.title("About")
    st.write("This app predicts NBA game outcomes.")    