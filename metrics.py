from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def calculate_metrics(name, method, results) -> dict:
    """
    Calcule les métriques de performance du modèle LLM.

    Args:
        results (pd.DataFrame):
            DataFrame retourné par la fonction `test()`.

    Returns:
        dict:
            Dictionnaire contenant les principales métriques de classification.
    """

    y_true = results["Attendue"]
    y_pred = results["Réponse"]

    metrics = {
        "Name": name,

        "Method": method,

        "Samples": len(results),

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision (macro)": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Recall (macro)": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "F1 (macro)": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Precision (weighted)": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall (weighted)": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1 (weighted)": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Temps moyen (s)": results["Temps (s)"].mean(),

        "Temps médian (s)": results["Temps (s)"].median(),

        "Temps P95 (s)": results["Temps (s)"].quantile(0.95),
    }

    return metrics

def print_metrics(metrics):
    for name, value in metrics.items():
        if "Temps" in name:
            print(f"{name} : {value:.3f} s")
        elif "Samples" in name:
            print(f"{name} : {value}")
        elif isinstance(value, str):
            print(f"{name} : {value}")
        else:
            print(f"{name} : {value:.2%}")

def report(results):
    report = classification_report(
        results["Attendue"],
        results["Réponse"],
        zero_division=0
    )

    print(report)

