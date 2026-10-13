import argparse
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import git
import optuna
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from processing import load_and_split, get_preprocessor, apply_smote

# --- THE OBJECTIVE FUNCTION (The "Judge") ---
def objective(trial, X_train, y_train, X_test, y_test, model_type):
    """This function is called by Optuna for every 'Trial'."""
    if model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 20)
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    elif model_type == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }
        model = RandomForestClassifier(**params, random_state=42)
    else:
        params = {"C": trial.suggest_float("C", 0.1, 10.0)}
        model = LogisticRegression(**params, max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # We want to maximize Recall for the fraud class ('1')
    report = classification_report(y_test, y_pred, output_dict=True)
    return report['1']['recall']

# ---THE MAIN TRAINING FUNCTION ---
def run_training(model_type, data_path, n_estimators, learning_rate, use_smote, tune):
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    # Get Data & Preprocess
    X_train, X_test, y_train, y_test = load_and_split(data_path)
    preprocessor = get_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    if use_smote:
        X_train_final, y_train_final = apply_smote(X_train_transformed, y_train)
        y_train_final = y_train_final.values
    else:
        X_train_final, y_train_final = X_train_transformed, y_train

    # START MLFLOW
    run_name = f"{model_type.upper()}_Tuning" if tune else f"{model_type.upper()}_Fixed"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit", commit_hash)
        # mlflow.log_param("use_smote", use_smote)

        if tune:
            print(f"🚀 Starting Hyperparameter Tuning for {model_type}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X_train_final, y_train_final, X_test_transformed, y_test, model_type), n_trials=10)
            
            # Log the winner
            best_params = study.best_params
            mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
            # mlflow.log_params(best_params)
            mlflow.log_metric("best_recall", study.best_value)
            
            # Re-train one last time with the winners
            if model_type == "xgboost": model = XGBClassifier(**best_params)
            elif model_type == "rf": model = RandomForestClassifier(**best_params)
            else: model = LogisticRegression(**best_params)
        else:
            # Standard single run with user parameters
            if model_type == "xgboost":
                model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, scale_pos_weight=10 if not use_smote else 1)
            elif model_type == "rf":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42)
            else:
                model = LogisticRegression(max_iter=1000)

        # Final Train & Evaluation
        model.fit(X_train_final, y_train_final)
        y_pred = model.predict(X_test_transformed)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        mlflow.log_metric("fraud_recall", report['1']['recall'])
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, f'{model_type}_model')
        
        print(f"✅ Success! Final Recall: {report['1']['recall']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--use_smote", type=str, default='False')
    parser.add_argument("--model_type", type=str, choices=["xgboost", "rf", "lr"], default="xgboost")
    parser.add_argument("--tune", type=str, default='False')
    args = parser.parse_args()
    
    run_training(args.model_type, args.data_path, args.n_estimators, args.learning_rate, args.use_smote.lower() == 'true', args.tune.lower() == 'true')
