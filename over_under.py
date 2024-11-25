import pandas as pd
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import logging

MODEL_FILE = 'nba_over_under_model.pkl'
FILE1 = 'nba_player_23_24.csv'
FILE2 = 'nba_player_24_25.csv'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def preprocess_data():
    logging.info("Loading datasets...")
    data1 = pd.read_csv(FILE1)
    data2 = pd.read_csv(FILE2)
    logging.info("Combining datasets...")
    combined_data = pd.concat([data1, data2], ignore_index=True)

    logging.info("Processing date columns...")
    combined_data['Year'] = pd.to_datetime(combined_data['Data']).dt.year
    combined_data['Month'] = pd.to_datetime(combined_data['Data']).dt.month
    combined_data['Day'] = pd.to_datetime(combined_data['Data']).dt.day

    prepared_data = combined_data[['Player', 'Tm', 'Opp', 'PTS', 'Year', 'Month', 'Day']]
    logging.info("Data preprocessing complete.")
    return prepared_data

def evaluate_model(prepared_data, thresholds):
    prepared_data = pd.get_dummies(prepared_data, columns=['Player', 'Tm', 'Opp'])
    
    for threshold in thresholds:
        logging.info(f"Training model for {threshold} points threshold...")
        X = prepared_data.drop(columns=['PTS'])
        y = (prepared_data['PTS'] > threshold).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logging.info(f"Model trained for {threshold} points. Accuracy: {accuracy:.2f}")

        print(f"\nClassification Report for {threshold} points:")
        print(classification_report(y_test, y_pred))

        # Feature Importance Chart
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = feature_importances.sort_values(ascending=False).head(25)
        top_features.plot(
            kind='bar', figsize=(10, 6), title=f"Top 25 Feature Importances for {threshold} Points"
        )
        plt.ylabel('Importance')
        plt.show()

        # Prediction Outcomes Chart
        results_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
        results_df['Correct'] = results_df['True'] == results_df['Predicted']
        correct_predictions = results_df[results_df['Correct'] == True].shape[0]
        incorrect_predictions = results_df[results_df['Correct'] == False].shape[0]

        plt.bar(['Correct', 'Incorrect'], [correct_predictions, incorrect_predictions], color=['green', 'red'])
        plt.title(f'Prediction Outcomes for {threshold} Points')
        plt.ylabel('Count')
        plt.show()

def predict_over_under(player, team1, team2, date, points):
    logging.info("Loading datasets...")
    data1 = pd.read_csv(FILE1)
    data2 = pd.read_csv(FILE2)
    combined_data = pd.concat([data1, data2], ignore_index=True)

    logging.info(f"Filtering data for player: {player}")
    player_data = combined_data[combined_data['Player'] == player]

    if player_data.empty:
        print(f"No data found for player {player}.")
        return

    logging.info("Processing date columns for prediction...")
    player_data['Year'] = pd.to_datetime(player_data['Data']).dt.year
    player_data['Month'] = pd.to_datetime(player_data['Data']).dt.month
    player_data['Day'] = pd.to_datetime(player_data['Data']).dt.day

    # Perform one-hot encoding
    player_data = pd.get_dummies(player_data, columns=['Tm', 'Opp'])

    # Prepare X and y
    X = player_data.drop(columns=['PTS', 'Player', 'Data'])
    y = (player_data['PTS'] > points).astype(int)

    # Ensure X contains only numeric data
    X = X.select_dtypes(include=['number'])

    # Train model
    model = RandomForestClassifier(random_state=42)
    logging.info("Training prediction model dynamically...")
    model.fit(X, y)

    # Create input data for prediction
    input_data = pd.DataFrame([{
        'Year': pd.to_datetime(date).year,
        'Month': pd.to_datetime(date).month,
        'Day': pd.to_datetime(date).day
    }])
    input_data[f'Tm_{team1}'] = 1
    input_data[f'Opp_{team2}'] = 1

    # Align input data with the model's feature set
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]

    # Ensure input_data contains only numeric data
    input_data = input_data.select_dtypes(include=['number'])

    # Make prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    predicted_class = prediction[0]
    probability = probabilities[0][predicted_class]

    logging.info(f"Prediction complete: {predicted_class} with confidence {probability:.2f}")
    print(f"Predicted {'Over' if predicted_class == 1 else 'Under'} for {points} points with confidence {probability:.2f}.")

def main():
    parser = argparse.ArgumentParser(description='NBA Over/Under Prediction Model')
    parser.add_argument('mode', choices=['evaluate', 'predict'], help='Mode to run: evaluate or predict')
    parser.add_argument('--player', help='Player name for prediction')
    parser.add_argument('--team1', help='Team 1 for prediction')
    parser.add_argument('--team2', help='Team 2 for prediction')
    parser.add_argument('--date', help='Date of the game for prediction')
    parser.add_argument('--points', type=int, help='Points threshold for prediction')

    args = parser.parse_args()

    if args.mode == 'evaluate':
        prepared_data = preprocess_data()
        evaluate_model(prepared_data, thresholds=[10, 20, 30])
    elif args.mode == 'predict':
        if not all([args.player, args.team1, args.team2, args.date, args.points]):
            print("Player, team1, team2, date, and points are required for prediction.")
            return
        predict_over_under(args.player, args.team1, args.team2, args.date, args.points)

if __name__ == '__main__':
    main()