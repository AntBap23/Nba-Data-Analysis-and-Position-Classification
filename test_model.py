import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def test_model_loading():
    """Test different methods to load the model"""
    print("Testing model loading methods...")
    
    # Method 1: Standard pickle
    try:
        with open("gradient_boosting_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✓ Standard pickle loading successful")
        print(f"Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"✗ Standard pickle failed: {e}")
    
    # Method 2: Joblib
    try:
        model = joblib.load("gradient_boosting_model.pkl")
        print("✓ Joblib loading successful")
        print(f"Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"✗ Joblib failed: {e}")
    
    # Method 3: Try with different protocols
    for protocol in [0, 1, 2, 3, 4, 5]:
        try:
            with open("gradient_boosting_model.pkl", "rb") as f:
                model = pickle.load(f)
            print(f"✓ Pickle protocol {protocol} successful")
            return model
        except Exception as e:
            print(f"✗ Pickle protocol {protocol} failed: {e}")
    
    return None

def create_new_model():
    """Create a new gradient boosting model if loading fails"""
    print("\nCreating new model from data...")
    
    try:
        # Load data
        df = pd.read_csv("data/player_logs_2023-24_with_bio.csv")
        print(f"Data loaded: {len(df)} rows")
        
        # Aggregate data by player
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['SEASON_ID', 'PLAYER_ID', 'GAME_ID', 'HEIGHT_IN', 'WEIGHT_LB']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        df_agg = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'POSITION'])[numeric_cols].mean().reset_index()
        
        # Clean position column
        df_agg['POSITION'] = df_agg['POSITION'].str.split('-').str[0].str.strip()
        
        # Prepare features and target
        X = df_agg.select_dtypes(include=[np.number]).drop('PLAYER_ID', axis=1)
        y = df_agg['POSITION']
        
        print(f"Features shape: {X.shape}")
        print(f"Target classes: {y.unique()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Test accuracy
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.3f}")
        
        # Save the new model
        joblib.dump(model, "gradient_boosting_model_new.pkl")
        print("✓ New model saved as gradient_boosting_model_new.pkl")
        
        return model, X.columns.tolist()
        
    except Exception as e:
        print(f"✗ Failed to create new model: {e}")
        return None, None

if __name__ == "__main__":
    model = test_model_loading()
    
    if model is None:
        print("\nModel loading failed. Creating new model...")
        model, feature_cols = create_new_model()
        if model is not None:
            print(f"New model created with {len(feature_cols)} features")
    else:
        print("Model loaded successfully!")
