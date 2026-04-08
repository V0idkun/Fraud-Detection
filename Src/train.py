import argparse
import mlflow
import mlflow.xgboost
import git
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from processing import load_and_split, get_preprocessor,apply_smote
from sklearn.metrics import accuracy_score

def run_training(model_type,data_path, n_estimators, learning_rate,use_smote):
    # 1. Start MLflow & Tag with Git
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    with mlflow.start_run(run_name=f"{model_type}_Fraud_Model"):
        mlflow.set_tag("git_commit", commit_hash)
        # mlflow.xgboost.autolog() # Automatically logs XGBoost params

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
            y_train_final = y_train_final.values
        else:
            X_train_final, y_train_final = X_train_transformed, y_train

        # 5. Build Pipeline with Class Imbalance fix
        # scale_pos_weight helps the model focus on the 1% of fraud cases

        if model_type == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBClassifier(
                n_estimators=n_estimators, 
                learning_rate=learning_rate,
                scale_pos_weight=10 if not use_smote else 1 # Adjust weight if SMOTE is on
            )
        elif model_type == "rf":
            mlflow.sklearn.autolog()
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            mlflow.sklearn.autolog()
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train_final,y_train_final)
        y_pred = model.predict(X_test_transformed)
        
        # pipeline = Pipeline([
        #     ('preprocessor', preprocessor),
        #     ('classifier', model)
        # ])

        # # 4. Train & Evaluate
        # pipeline.fit(X_train_final, y_train_final)
        # y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)


        # 5. Log specific Fraud Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("fraud_recall", report['1']['recall'])
        # mlflow.sklearn.log_model(pipeline, "model_pipeline")
        mlflow.sklearn.log_model(model, f'{model_type}_model')
        
        print(f"✅ Success! Recall: {report['1']['recall']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--use_smote", action='store_true', help="Whether to apply SMOTE for class imbalance")
    parser.add_argument("--model_type", type=str, choices=["xgboost", "rf", "lr"], help="Type of model to train")
    args = parser.parse_args()
    
    run_training(args.model_type, args.data_path, args.n_estimators, args.learning_rate, args.use_smote)