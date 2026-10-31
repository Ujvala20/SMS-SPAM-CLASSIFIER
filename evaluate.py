import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve

METRICS = ["accuracy", "precision", "recall", "f1"]


def load_results(path="results.json"):
    with open(path) as f:
        return json.load(f)


def load_predictions(path="test_predictions.csv"):
    return pd.read_csv(path)


def metrics_comparison(results):
    rows = []
    for model, m in results.items():
        for metric in METRICS:
            rows.append({"model": model, "metric": metric, "score": m[metric]})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="metric", y="score", color="model", barmode="group",
                 title="Model comparison", range_y=[0.5, 1.0])
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return fig


def confusion_matrix_fig(results, model):
    cm = results[model]["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted ham", "Predicted spam"],
                    y=["Actual ham", "Actual spam"],
                    title=f"Confusion matrix: {model}")
    return fig


def roc_fig(preds):
    fig = go.Figure()
    for col in preds.columns[2:]:
        fpr, tpr, _ = roc_curve(preds["y_true"], preds[col])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=col))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate",
                      legend=dict(orientation="h", y=-0.3))
    return fig


def pr_fig(preds):
    fig = go.Figure()
    for col in preds.columns[2:]:
        p, r, _ = precision_recall_curve(preds["y_true"], preds[col])
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=col))
    fig.update_layout(title="Precision-recall curve", xaxis_title="Recall",
                      yaxis_title="Precision",
                      legend=dict(orientation="h", y=-0.3))
    return fig


if __name__ == "__main__":
    data = load_results()
    results, best = data["results"], data["best_model"]
    preds = load_predictions()

    metrics_comparison(results).show()
    confusion_matrix_fig(results, best).show()
    roc_fig(preds).show()
    pr_fig(preds).show()