import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NBA Player Analysis & Classification",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B35;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the NBA data"""
    try:
        df = pd.read_csv("data/player_logs_2023-24_with_bio.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the gradient boosting model"""
    # Try to load the new model first, then fall back to original
    model_files = ["gradient_boosting_model_new.pkl", "gradient_boosting_model.pkl"]
    
    for model_file in model_files:
        try:
            # Try joblib first (recommended for sklearn models)
            import joblib
            model = joblib.load(model_file)
            st.success(f"Model loaded successfully from {model_file}")
            return model
        except:
            try:
                # Fall back to pickle
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                st.success(f"Model loaded successfully from {model_file}")
                return model
            except Exception as e:
                st.warning(f"Could not load {model_file}: {e}")
                continue
    
    # If all loading attempts fail, create a new model
    st.warning("Could not load existing model. Creating new model from data...")
    return create_new_model()

def create_new_model():
    """Create a new gradient boosting model if loading fails"""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        import joblib
        
        # Load and prepare data
        df = load_data()
        if df is None:
            return None
            
        df_agg = prepare_data_for_model(df)
        
        # Prepare features and target
        X = df_agg.select_dtypes(include=[np.number]).drop('PLAYER_ID', axis=1)
        y = df_agg['POSITION']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Save the new model
        joblib.dump(model, "gradient_boosting_model_new.pkl")
        
        accuracy = model.score(X_test, y_test)
        st.success(f"New model created with {accuracy:.1%} accuracy")
        
        return model
        
    except Exception as e:
        st.error(f"Failed to create new model: {e}")
        return None

def prepare_data_for_model(df):
    """Prepare aggregated data for model prediction"""
    # Aggregate player stats by season
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove ID columns and other non-stat columns
    exclude_cols = ['SEASON_ID', 'PLAYER_ID', 'GAME_ID', 'HEIGHT_IN', 'WEIGHT_LB']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Group by player and aggregate
    df_agg = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'POSITION'])[numeric_cols].mean().reset_index()
    
    # Clean position column to keep only first position
    df_agg['POSITION'] = df_agg['POSITION'].str.split('-').str[0].str.strip()
    
    return df_agg

def predict_position(model, player_stats, feature_columns):
    """Predict player position using the model"""
    try:
        # Check if model has prediction methods
        if not hasattr(model, 'predict'):
            st.error("Model does not have prediction capability.")
            return None, None, None
        
        # Ensure we have the right features in the right order
        stats_df = pd.DataFrame([player_stats])
        stats_df = stats_df.reindex(columns=feature_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(stats_df)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(stats_df)[0]
        else:
            # Fallback: create dummy probabilities
            probabilities = [1.0, 0.0, 0.0]  # Assume high confidence for predicted class
        
        # Get classes if available
        if hasattr(model, 'classes_'):
            classes = model.classes_
        else:
            # Fallback: use known NBA positions
            classes = ['Center', 'Forward', 'Guard']
        
        return prediction, probabilities, classes
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏀 NBA Player Analysis & Classification</h1>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    if df is None or model is None:
        st.error("Failed to load data or model. Please check file paths.")
        return
    
    # Prepare aggregated data
    df_agg = prepare_data_for_model(df)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["📊 Data Overview", 
                                "🔍 Player Analysis", 
                                "🤖 Position Prediction", 
                                "🎯 Model Insights"])
    
    if page == "📊 Data Overview":
        show_data_overview(df, df_agg)
    elif page == "🔍 Player Analysis":
        show_player_analysis(df, df_agg)
    elif page == "🤖 Position Prediction":
        show_position_prediction(model, df_agg)
    elif page == "🎯 Model Insights":
        show_model_insights(model, df_agg)

def show_data_overview(df, df_agg):
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", f"{len(df):,}")
    with col2:
        st.metric("Unique Players", f"{df['PLAYER_ID'].nunique():,}")
    with col3:
        st.metric("Teams", f"{df['TEAM'].nunique()}")
    with col4:
        st.metric("Season", "2023-24")
    
    # Position distribution
    st.subheader("Position Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        pos_counts = df_agg['POSITION'].value_counts()
        fig = px.pie(values=pos_counts.values, names=pos_counts.index, 
                     title="Player Distribution by Position")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(x=pos_counts.index, y=pos_counts.values,
                     title="Number of Players by Position",
                     labels={'x': 'Position', 'y': 'Number of Players'})
        fig.update_traces(marker_color='#FF6B35')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.subheader("Sample Player Data")
    st.dataframe(df_agg.head(10), use_container_width=True)
    
    # Basic statistics
    st.subheader("Key Statistics Summary")
    key_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FT_PCT']
    available_stats = [stat for stat in key_stats if stat in df_agg.columns]
    
    if available_stats:
        stats_summary = df_agg[available_stats].describe()
        st.dataframe(stats_summary, use_container_width=True)

def show_player_analysis(df, df_agg):
    st.markdown('<h2 class="sub-header">🔍 Individual Player Analysis</h2>', unsafe_allow_html=True)
    
    # Player selection
    players = sorted(df_agg['PLAYER_NAME'].unique())
    selected_player = st.selectbox("Select a player:", players)
    
    if selected_player:
        player_data = df_agg[df_agg['PLAYER_NAME'] == selected_player].iloc[0]
        player_games = df[df['PLAYER_NAME'] == selected_player]
        
        # Player info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Position", player_data['POSITION'])
        with col2:
            if 'PTS' in player_data:
                st.metric("Avg Points", f"{player_data['PTS']:.1f}")
        with col3:
            if 'REB' in player_data:
                st.metric("Avg Rebounds", f"{player_data['REB']:.1f}")
        with col4:
            if 'AST' in player_data:
                st.metric("Avg Assists", f"{player_data['AST']:.1f}")
        
        # Performance over time
        if len(player_games) > 1:
            st.subheader("Performance Over Season")
            
            # Convert game date to datetime
            player_games_copy = player_games.copy()
            player_games_copy['GAME_DATE'] = pd.to_datetime(player_games_copy['GAME_DATE'])
            player_games_copy = player_games_copy.sort_values('GAME_DATE')
            
            # Plot key stats over time
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=['Points', 'Rebounds', 'Assists', 'Field Goal %'])
            
            if 'PTS' in player_games_copy.columns:
                fig.add_trace(go.Scatter(x=player_games_copy['GAME_DATE'], y=player_games_copy['PTS'],
                                       mode='lines+markers', name='Points'), row=1, col=1)
            
            if 'REB' in player_games_copy.columns:
                fig.add_trace(go.Scatter(x=player_games_copy['GAME_DATE'], y=player_games_copy['REB'],
                                       mode='lines+markers', name='Rebounds'), row=1, col=2)
            
            if 'AST' in player_games_copy.columns:
                fig.add_trace(go.Scatter(x=player_games_copy['GAME_DATE'], y=player_games_copy['AST'],
                                       mode='lines+markers', name='Assists'), row=2, col=1)
            
            if 'FG_PCT' in player_games_copy.columns:
                fig.add_trace(go.Scatter(x=player_games_copy['GAME_DATE'], y=player_games_copy['FG_PCT'],
                                       mode='lines+markers', name='FG%'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Player comparison with position average
        st.subheader("Comparison with Position Average")
        position_avg = df_agg[df_agg['POSITION'] == player_data['POSITION']].select_dtypes(include=[np.number]).mean()
        
        comparison_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT']
        available_comparison = [stat for stat in comparison_stats if stat in player_data.index and stat in position_avg.index]
        
        if available_comparison:
            player_values = [player_data[stat] for stat in available_comparison]
            position_values = [position_avg[stat] for stat in available_comparison]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name=selected_player, x=available_comparison, y=player_values))
            fig.add_trace(go.Bar(name=f'{player_data["POSITION"]} Average', x=available_comparison, y=position_values))
            fig.update_layout(title=f"{selected_player} vs {player_data['POSITION']} Average",
                            barmode='group')
            st.plotly_chart(fig, use_container_width=True)

def show_position_prediction(model, df_agg):
    st.markdown('<h2 class="sub-header">🤖 Position Prediction</h2>', unsafe_allow_html=True)
    
    st.write("Enter player statistics to predict their position:")
    
    # Get feature columns (numeric columns excluding IDs and target)
    numeric_cols = df_agg.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['PLAYER_ID']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create input form
    col1, col2 = st.columns(2)
    
    player_stats = {}
    
    # Common basketball stats for input
    key_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'MIN']
    available_key_stats = [stat for stat in key_stats if stat in feature_cols]
    
    with col1:
        st.subheader("Primary Stats")
        for i, stat in enumerate(available_key_stats[:4]):
            if stat in df_agg.columns:
                avg_val = df_agg[stat].mean()
                player_stats[stat] = st.number_input(f"{stat}", value=float(avg_val), step=0.1)
    
    with col2:
        st.subheader("Additional Stats")
        for i, stat in enumerate(available_key_stats[4:]):
            if stat in df_agg.columns:
                avg_val = df_agg[stat].mean()
                player_stats[stat] = st.number_input(f"{stat}", value=float(avg_val), step=0.1)
    
    # Fill remaining features with average values
    for col in feature_cols:
        if col not in player_stats:
            player_stats[col] = df_agg[col].mean()
    
    if st.button("Predict Position", type="primary"):
        prediction, probabilities, classes = predict_position(model, player_stats, feature_cols)
        
        if prediction is not None:
            st.success(f"Predicted Position: **{prediction}**")
            
            # Show probabilities
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Position': classes,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(prob_df, x='Position', y='Probability',
                        title="Position Prediction Probabilities")
            fig.update_traces(marker_color='#FF6B35')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top probabilities as metrics
            col1, col2, col3 = st.columns(3)
            for i, (pos, prob) in enumerate(zip(prob_df['Position'][:3], prob_df['Probability'][:3])):
                with [col1, col2, col3][i]:
                    st.metric(f"{i+1}. {pos}", f"{prob:.1%}")


def show_model_insights(model, df_agg):
    st.markdown('<h2 class="sub-header">🎯 Model Insights</h2>', unsafe_allow_html=True)
    
    # Check if model is properly loaded
    if model is None:
        st.error("Model not loaded properly. Please check the model file.")
        return
    
    # Model information
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Model Type:** {type(model).__name__}")
        
        # Handle case where model might not have classes_ attribute
        if hasattr(model, 'classes_'):
            st.info(f"**Number of Classes:** {len(model.classes_)}")
            st.info(f"**Classes:** {', '.join(model.classes_)}")
        else:
            # Fallback: get classes from the data
            unique_positions = df_agg['POSITION'].unique()
            st.info(f"**Number of Classes:** {len(unique_positions)}")
            st.info(f"**Classes:** {', '.join(sorted(unique_positions))}")
    
    with col2:
        if hasattr(model, 'n_estimators'):
            st.info(f"**Number of Estimators:** {model.n_estimators}")
        elif hasattr(model, 'n_estimators_'):
            st.info(f"**Number of Estimators:** {model.n_estimators_}")
        else:
            st.info("**Number of Estimators:** Not available")
            
        if hasattr(model, 'max_depth'):
            st.info(f"**Max Depth:** {model.max_depth}")
        elif hasattr(model, 'max_depth_'):
            st.info(f"**Max Depth:** {model.max_depth_}")
        else:
            st.info("**Max Depth:** Not available")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        # Get feature names
        numeric_cols = df_agg.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != 'PLAYER_ID']
        
        try:
            if len(model.feature_importances_) == len(feature_cols):
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(importance_df, x='Importance', y='Feature',
                            orientation='h', title="Top 15 Most Important Features")
                fig.update_traces(marker_color='#FF6B35')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance data doesn't match the expected number of features.")
        except Exception as e:
            st.warning(f"Could not display feature importance: {e}")
    else:
        st.info("Feature importance not available for this model type.")
    
    # Position distribution in dataset
    st.subheader("Position Distribution in Training Data")
    pos_dist = df_agg['POSITION'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=pos_dist.values, names=pos_dist.index,
                     title="Training Data Position Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Position Counts:**")
        for pos, count in pos_dist.items():
            st.write(f"• {pos}: {count} players")

if __name__ == "__main__":
    main()
