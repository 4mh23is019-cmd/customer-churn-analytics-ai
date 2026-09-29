"""
Customer Churn Analytics & AI Prediction
Data Analytics & AI Internship Project
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

def create_dataset(n=1200):
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(18, 71, n)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(20, 130, n), 2)
    calls = rng.poisson(1.7, n).clip(0, 8)
    satisfaction = rng.integers(1, 11, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55,.27,.18])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35,.48,.17])
    payment = rng.choice(["Electronic check", "Card", "Bank transfer"], n, p=[.42,.35,.23])
    support = rng.choice(["Yes", "No"], n, p=[.32,.68])
    score = (-1.7 + 1.35*(contract=="Month-to-month") + .55*(internet=="Fiber optic")
             + .10*calls - .10*satisfaction + .012*monthly - .018*tenure
             + .35*(payment=="Electronic check") - .45*(contract=="Two year")
             - .25*(support=="Yes") + rng.normal(0,.65,n))
    prob = 1/(1+np.exp(-score))
    churn = rng.binomial(1, prob)
    df = pd.DataFrame({
        "Age":age, "TenureMonths":tenure, "MonthlyCharges":monthly,
        "SupportCalls":calls, "SatisfactionScore":satisfaction,
        "Contract":contract, "InternetService":internet,
        "PaymentMethod":payment, "TechSupport":support, "Churn":churn})
    for col in ["MonthlyCharges","SatisfactionScore"]:
        idx = rng.choice(df.index, size=int(.02*n), replace=False)
        df.loc[idx,col] = np.nan
    return df

def train_and_analyze(df):
    numeric = ["Age","TenureMonths","MonthlyCharges","SupportCalls","SatisfactionScore"]
    categorical = ["Contract","InternetService","PaymentMethod","TechSupport"]
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),
                          ("scaler",StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                          ("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = RandomForestClassifier(n_estimators=250,max_depth=10,
                                   random_state=RANDOM_STATE,class_weight="balanced")
    pipe = Pipeline([("preprocessor",pre),("model",model)])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.20,random_state=RANDOM_STATE,stratify=y)
    pipe.fit(Xtr,ytr)
    pred = pipe.predict(Xte)
    metrics = {
        "Accuracy":accuracy_score(yte,pred),
        "Precision":precision_score(yte,pred,zero_division=0),
        "Recall":recall_score(yte,pred,zero_division=0),
        "F1 Score":f1_score(yte,pred,zero_division=0)}
    return pipe, yte, pred, metrics

def charts(df, pipe, yte, pred):
    plt.figure(figsize=(7,5)); df["Churn"].value_counts().sort_index().plot(kind="bar")
    plt.xticks([0,1],["Stayed","Churned"],rotation=0); plt.ylabel("Customers")
    plt.title("Customer Churn Distribution"); plt.tight_layout()
    plt.savefig(OUTPUT/"churn_distribution.png",dpi=160); plt.close()

    plt.figure(figsize=(8,5)); df.groupby("Contract")["Churn"].mean().mul(100).sort_values().plot(kind="bar")
    plt.ylabel("Churn Rate (%)"); plt.title("Churn Rate by Contract Type"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(OUTPUT/"churn_by_contract.png",dpi=160); plt.close()

    plt.figure(figsize=(8,5)); df.groupby("SatisfactionScore")["Churn"].mean().mul(100).plot(marker="o")
    plt.xlabel("Satisfaction Score"); plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate vs Satisfaction"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(OUTPUT/"churn_vs_satisfaction.png",dpi=160); plt.close()

    cm = confusion_matrix(yte,pred)
    plt.figure(figsize=(6,5)); plt.imshow(cm); plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0,1],["Stayed","Churned"]); plt.yticks([0,1],["Stayed","Churned"])
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
    plt.tight_layout(); plt.savefig(OUTPUT/"confusion_matrix.png",dpi=160); plt.close()

    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_,index=names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(9,5)); imp.sort_values().plot(kind="barh")
    plt.xlabel("Importance"); plt.title("Top Predictive Features"); plt.tight_layout()
    plt.savefig(OUTPUT/"feature_importance.png",dpi=160); plt.close()

def main():
    df = create_dataset()
    df.to_csv(OUTPUT/"customer_churn_dataset.csv",index=False)
    pipe,yte,pred,metrics = train_and_analyze(df)
    charts(df,pipe,yte,pred)
    pd.DataFrame([metrics]).to_csv(OUTPUT/"model_metrics.csv",index=False)
    with open(OUTPUT/"summary.txt","w",encoding="utf-8") as f:
        f.write("Customer Churn Analytics & AI Prediction\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Churn rate: {df.Churn.mean()*100:.2f}%\n")
        for k,v in metrics.items(): f.write(f"{k}: {v:.4f}\n")
    print(f"Rows: {len(df)} | Churn rate: {df.Churn.mean()*100:.2f}%")
    for k,v in metrics.items(): print(f"{k}: {v:.3f}")
    print("Project outputs created in ./output")

if __name__ == "__main__":
    main()
