import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

def load_and_prep_data(filepath='data/features_mafaulda.csv'):
    df = pd.read_csv(filepath)
    y = df['fault_type']
    cols_to_drop = ['fault_family', 'fault_type', 'severity', 'severity_unit', 'bearing_position']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    X = X.select_dtypes(include=['number']).fillna(X.select_dtypes(include=['number']).mean())
    return X, y 

def train_and_log_model(X_train, X_test, y_train, y_test, model, model_name, run_name, params):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("fallas-maquinaria")
    
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f1_mac = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        
        classes = list(model.classes_)
        if 'normal' in classes:
            idx_normal = classes.index('normal')
            recalls = recall_score(y_test, y_pred, average=None)
            rec_normal = recalls[idx_normal]
        else:
            rec_normal = 0.0
            
        mlflow.log_param("modelo", model_name)
        mlflow.log_param("tipo_particion", "estratificada 80/20")
        mlflow.log_param("version_datos", "v1 solo tiempo")
        for key, value in params.items():
            mlflow.log_param(key, value)
            
        mlflow.log_metric("f1_macro", f1_mac)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall_normal", rec_normal)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation='vertical')
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.tight_layout()
        
        artifact_path = f"matriz_confusion_{model_name.replace(' ', '_')}.png"
        plt.savefig(artifact_path)
        mlflow.log_artifact(artifact_path)
        plt.close()
        
        mlflow.sklearn.log_model(model, name="modelo", skops_trusted_types=["sklearn.tree._tree.Tree"])
        print(f" {run_name} registrado. F1 Macro: {f1_mac:.4f} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    print("Cargando datos...")
    X, y = load_and_prep_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Entrenando Regresión Logística (Línea Base)...")
    lr_params = {'max_iter': 1000, 'C': 1.0}
    lr_model = LogisticRegression(**lr_params, random_state=42)
    train_and_log_model(X_train_scaled, X_test_scaled, y_train, y_test, lr_model, "Regresion Logistica", "lr_estratificado_base", lr_params)
    
    print("Entrenando Random Forest...")
    rf_params = {'n_estimators': 200, 'max_depth': 15}
    rf_model = RandomForestClassifier(**rf_params, random_state=42)
    train_and_log_model(X_train, X_test, y_train, y_test, rf_model, "Random Forest", "rf_estratificado_n200", rf_params)