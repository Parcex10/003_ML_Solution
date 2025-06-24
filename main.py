import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from utils import compute_rsi

def load_data(path):
    df = pd.read_csv(path)
    return df.dropna()

def add_technical_indicators(df):
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EWMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["Volatility_5"] = df["Close"].rolling(window=5).std()
    df["RSI_14"] = compute_rsi(df["Close"])
    df["Volume_Change"] = df["Volume"].pct_change()
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_upper"] = rolling_mean + 2 * rolling_std
    df["BB_lower"] = rolling_mean - 2 * rolling_std
    return df.dropna()

def build_pipeline(numerical_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
        ]
    )
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("pca", PCA(n_components=6)),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(max_iter=1000, solver='liblinear'))
    ])
    return pipeline

def main():
    # Rutas
    data_path = "../data/SPY_dataset_project.csv"
    
    # Cargar y procesar datos
    df = load_data(data_path)
    df = add_technical_indicators(df)
    
    # Variables
    numerical = ["Close", "Volume", "return", "SMA_5", "SMA_10", "EWMA_5", "Volatility_5",
                 "RSI_14", "Volume_Change", "BB_upper", "BB_lower"]
    categorical = ["market_sentiment", "recession_expectation", "growing_sector",
                   "investor_type", "news_impact", "policy_uncertainty"]
    
    X = df[numerical + categorical]
    y = df["signal"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        stratify=y, random_state=42)
    
    # Pipeline y GridSearch
    pipeline = build_pipeline(numerical, categorical)
    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    
    # Entrenar
    search.fit(X_train, y_train)
    print("Mejores parámetros:", search.best_params_)
    
    # Evaluación
    print("\n--- Entrenamiento ---")
    print(classification_report(y_train, search.predict(X_train)))
    
    print("\n--- Prueba ---")
    print(classification_report(y_test, search.predict(X_test)))

if __name__ == "__main__":
    main()
