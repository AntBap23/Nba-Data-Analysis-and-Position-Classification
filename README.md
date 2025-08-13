# 🏀 NBA Data Analysis and Position Classification

A comprehensive machine learning project for analyzing NBA player statistics and predicting player positions using gradient boosting classification. This project includes data collection, processing, analysis, and an interactive Streamlit web application for real-time predictions and insights.

## 📊 Project Overview

This project analyzes NBA player performance data from the 2023-24 season to:
- **Analyze player statistics** and performance trends
- **Predict player positions** (Guard, Forward, Center) using machine learning
- **Provide interactive visualizations** through a web application
- **Offer insights** into player performance and model behavior

## 🚀 Features

### Interactive Streamlit Web Application
- **📊 Data Overview**: Dataset statistics, position distributions, and key metrics
- **🔍 Player Analysis**: Individual player performance metrics and season trends
- **🤖 Position Prediction**: Real-time position classification with probability scores
- **🎯 Model Insights**: Feature importance analysis and model transparency

### Machine Learning Model
- **Gradient Boosting Classifier** with 78.9% accuracy
- **22 statistical features** including points, rebounds, assists, shooting percentages
- **3 position classes**: Guard, Forward, Center
- **Robust preprocessing** with data aggregation and feature engineering

### Data Pipeline
- **NBA API integration** for real-time data collection
- **Player biographical data** merging for enhanced features
- **23,770 game log entries** with comprehensive statistics
- **448 unique players** from the 2023-24 season

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Nba-Data-Analysis-and-Position-Classification.git
   cd Nba-Data-Analysis-and-Position-Classification
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Run the Streamlit Web Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`

### Explore the Jupyter Notebooks
- **`model.ipynb`**: Model development and training
- **`player_logs.ipynb`**: Exploratory data analysis

### Data Collection Scripts
```bash
# Fetch player game logs
python scripts/fetch_one_season_logs.py

# Fetch player biographical data
python scripts/fetch_player_bio.py

# Merge datasets
python merge_bio_into_logs.py
```

## 📁 Project Structure

```
├── app.py                              # Streamlit web application
├── model.ipynb                         # Model development notebook
├── player_logs.ipynb                   # Data analysis notebook
├── requirements.txt                    # Python dependencies
├── gradient_boosting_model_new.pkl     # Trained ML model
├── test_model.py                       # Model testing utilities
├── merge_bio_into_logs.py              # Data merging script
├── data/                               # Dataset directory
│   ├── player_logs_2023-24_with_bio.csv
│   ├── nba_player_bio.csv
│   └── player_logs_with_position_2023-2024.csv
└── scripts/                            # Data collection scripts
    ├── fetch_one_season_logs.py
    ├── fetch_player_bio.py
    └── merge_positions.py
```

## 🔬 Model Details

### Features Used
- **Shooting**: FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT
- **Rebounding**: OREB, DREB, REB
- **Playmaking**: AST, TOV
- **Defense**: STL, BLK
- **Efficiency**: PTS, PLUS_MINUS, MIN
- **Advanced**: Various calculated ratios and per-game averages

### Model Performance
- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 78.9%
- **Training Data**: 358 players
- **Test Data**: 90 players
- **Cross-validation**: 80/20 train-test split

### Position Distribution
- **Guards**: ~40% of players
- **Forwards**: ~45% of players  
- **Centers**: ~15% of players

## 📈 Key Insights

1. **Most Important Features**: Points per game, rebounds, assists, and shooting efficiency are the strongest predictors of position
2. **Position Characteristics**:
   - **Guards**: High assists, three-point shooting, lower rebounding
   - **Forwards**: Balanced statistics, versatile skill sets
   - **Centers**: High rebounding, blocks, field goal percentage
3. **Model Accuracy**: 78.9% accuracy demonstrates strong predictive capability while acknowledging the evolving nature of basketball positions

## 🛠️ Technologies Used

- **Python 3.12**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **pandas & numpy**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **matplotlib & seaborn**: Statistical plotting
- **nba_api**: NBA data collection
- **joblib**: Model serialization

## 📊 Data Sources

- **NBA API**: Official NBA statistics and game logs
- **Player Biographical Data**: Height, weight, birthdate, team information
- **Season Coverage**: Complete 2023-24 NBA regular season

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NBA API** for providing comprehensive basketball statistics
- **scikit-learn** community for excellent machine learning tools
- **Streamlit** for making web app development accessible
- **Basketball analytics community** for inspiration and methodologies

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out through GitHub.

---

**⭐ If you found this project helpful, please consider giving it a star!**