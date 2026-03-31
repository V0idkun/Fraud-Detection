import argparse
import mlflow
import mlflow.xgboost
import git
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from processing import load_and_split, get_preprocessor

def run_training(data_path, n_estimators, learning_rate):
    # 1. Start MLflow & Tag with Git
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    with mlflow.start_run(run_name="XGBoost_Fraud_Model"):
        mlflow.set_tag("git_commit", commit_hash)
        mlflow.xgboost.autolog() # Automatically logs XGBoost params

        # 2. Get Data
        X_train, X_test, y_train, y_test = load_and_split(data_path)
        preprocessor = get_preprocessor(X_train)

        # 3. Build Pipeline with Class Imbalance fix
        # scale_pos_weight helps the model focus on the 1% of fraud cases
        model = XGBClassifier(
            n_estimators=n_estimators, 
            learning_rate=learning_rate,
            scale_pos_weight=10, # Rough estimate for 2% fraud 
            use_label_encoder=False
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # 4. Train & Evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # 5. Log specific Fraud Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("fraud_recall", report['1']['recall'])
        mlflow.sklearn.log_model(pipeline, "model_pipeline")
        
        print(f"✅ Success! Recall: {report['1']['recall']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--learning_rate", type=float)
    args = parser.parse_args()
    
    run_training(args.data_path, args.n_estimators, args.learning_rate)