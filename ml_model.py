import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_csv("placementdata.csv")
    
    #Drop StudentID as it is not a predictive feature
    if 'StudentID' in df.columns:
        df = df.drop('StudentID', axis=1)

    #Convert Categorical "Yes/No" to 1/0 (Binary)
    df['ExtracurricularActivities'] = df['ExtracurricularActivities'].map({'Yes': 1, 'No': 0})
    df['PlacementTraining'] = df['PlacementTraining'].map({'Yes': 1, 'No': 0})
    df['PlacementStatus'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})

    X = df.drop('PlacementStatus', axis=1)
    y = df['PlacementStatus']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    

    accuracy = model.score(X_test, y_test)
    print(f"Model trained successfully with an accuracy of: {accuracy * 100:.2f}%")

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_and_save_model()