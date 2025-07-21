import gradio as gr
import pandas as pd
import joblib
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load model

model = joblib.load(
    "/customer_churn_predict/modelsrc/logistic_model.pkl"
)


# Wrangling function (from above)
def data_wrangle(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.lower().str.replace(" ", "_")
        # totalcharge column is string, we need to convert it to float
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

        df["multiplelines"] = df["multiplelines"].replace("no_phone_service", "no")
        condition_list = [
            (df["phoneservice"] == "yes") & (df["multiplelines"] == "no"),
            (df["internetservice"].isin(["fiber_optic", "dsl"]))
            & (df["multiplelines"] == "no"),
            df["multiplelines"] == "yes",
        ]
        choice_list = ["phone service only", "internet service only", "multiplelines"]

        df["service_line"] = np.select(condition_list, choice_list, default="Unknown")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df.drop(
        columns=["totalcharges", "phoneservice", "internetservice", "multiplelines"],
        inplace=True,
        axis=1,
    )

    binary_map = {"yes": "1", "no": "0", "male": "1", "female": "0"}
    for col in ["gender", "partner", "dependents"]:
        df[col] = df[col].map(binary_map)
        df[col] = df[col].astype(int)

    # prepare features for engineering
    numerical_cols = ["tenure", "monthlycharges"]
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    feature_processor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return feature_processor


# Prediction function
def predict_churn(file):
    df = pd.read_csv(file.name)

    # Store Customer ID
    customer_ids = df["customerID"] if "customerID" in df.columns else df.index

    # Wrangle and drop customerID before predict
    df_processed = data_wrangle(df)
    X = df_processed.drop(columns=["customerid"], errors="ignore")
    feature_processor = feature_engineering(X)
    X_transformed = feature_processor.fit_transform(X)

    y_pred = model.predict(X_transformed)

    # Prepare result
    result_df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "Prediction": ["Churn" if y == 1 else "Not Churn" for y in y_pred],
        }
    )

    # Summary
    churn_count = sum(y_pred)
    not_churn_count = len(y_pred) - churn_count
    summary = f"🔍 Churn: {churn_count} | ✅ Not Churn: {not_churn_count} | 📊 Total: {len(y_pred)}"

    # Save CSV result
    result_df.to_csv("prediction_result.csv", index=False)

    return summary, result_df, "prediction_result.csv"


# Gradio UI
interface = gr.Interface(
    fn=predict_churn,
    inputs=gr.File(label="Upload CSV file (customer info)"),
    outputs=[
        gr.Text(label="Prediction Summary"),
        gr.Dataframe(label="Prediction Table"),
        gr.File(label="Download CSV"),
    ],
    title="Customer Churn Prediction App",
    description="Upload a CSV file to predict which customers are likely to churn.",
)

interface.launch(share=True)
