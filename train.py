import argparse
import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import  StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


if __name__ == "__main__":

    ### MLFLOW Experiment setup
    experiment_name="appointment_cancellation_detector"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    client = mlflow.tracking.MlflowClient()
    # No client.create_run() here: start_run() below creates the run,
    # or reuses the one prepared by `mlflow run` (MLproject).

    print("training model...")
    
    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False) # We won't log models right away

    # Parse arguments given in shell script
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    # Import dataset
    df = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/doctolib_simplified_dataset_01.csv")

    # X, y split 
    X = df.iloc[:, 3:-1]
    y = df.iloc[:, -1].apply(lambda x: 0 if x=="No" else 1)

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # Preprocessing 
    def date_processing(df):
        df = df.copy()

        ## Transform datetime into a number
        df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], yearfirst=True)
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], yearfirst=True)

        ## Get the difference between scheduled day and appointment
        df["time_difference_between_scheduled_and_appointment"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

        ## Remove redundant info 
        df = df.drop(["ScheduledDay", "AppointmentDay"], axis=1)

        return df 

    date_preprocessor = FunctionTransformer(date_processing)

    # Preprocessing 
    categorical_features = ["Gender", "Neighbourhood"] # Select all the columns containing strings
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    numerical_feature_mask = ~X_train.columns.isin(["Gender", "Neighbourhood", "ScheduledDay","AppointmentDay"]) # Select all the columns containing anything else than strings
    numerical_features = X_train.columns[numerical_feature_mask]
    numerical_transformer = StandardScaler()

    feature_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical_transformer", categorical_transformer, categorical_features),
            ("numerical_transformer", numerical_transformer, numerical_features)
        ]
    )

    # Pipeline 
    n_estimators = int(args.n_estimators)
    min_samples_split=int(args.min_samples_split)

    model = Pipeline(steps=[
        ("Dates_preprocessing", date_preprocessor),
        ('features_preprocessing', feature_preprocessor),
        ("Regressor",RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split))
    ])

    # Log experiment to MLFlow
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)

        # Log model seperately to have more flexibility on setup 
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="appointment_cancellation_detector",
            registered_model_name="appointment_cancellation_detector_RF",
            signature=infer_signature(X_train, predictions)
        )

        # Put the "challenger" alias on the new version (as in Medhi's train_repo)
        client.set_registered_model_alias(
            name="appointment_cancellation_detector_RF",
            alias="challenger",
            version=model_info.registered_model_version,
        )
        print(f"[INFO] Model version {model_info.registered_model_version} -> alias 'challenger'")
        
    print("...Done!")
    print(f"---Total training time: {time.time()-start_time}")