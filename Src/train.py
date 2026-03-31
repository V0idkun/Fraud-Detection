import argparse
import mlflow
import mlflow.xgboost
import git
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from processing import load_and_split, get_preprocessor,apply_smote
from sklearn.metrics import accuracy_score

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

        # 3. Transform the data FIRST (SMOTE needs numbers, not 'Rent'/'Own' strings)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
    
        # 4. Apply SMOTE if requested
        if use_smote:
            mlflow.log_param("sampling_strategy", "SMOTE")
            X_train_final, y_train_final = apply_smote(X_train_transformed, y_train)
        else:
            X_train_final, y_train_final = X_train_transformed, y_train

        # 5. Build Pipeline with Class Imbalance fix
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
        pipeline.fit(X_train_final, y_train_final)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)


        # 5. Log specific Fraud Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("fraud_recall", report['1']['recall'])
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model_pipeline")
        
        print(f"✅ Success! Recall: {report['1']['recall']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--use_smote", action='store_true', help="Whether to apply SMOTE for class imbalance")
    args = parser.parse_args()
    
    run_training(args.data_path, args.n_estimators, args.learning_rate, args.use_smote)