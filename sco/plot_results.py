from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

project_dir=Path(__file__).resolve().parent
matrix_df=pd.read_csv(
    project_dir/"confusion_matrix.csv",
    index_col=0
)

matrix=matrix_df.to_numpy()

fig,ax=plt.subplots(figsize=(6,5))
image=ax.imshow(matrix,cmap="Blues")

ax.set_xticks(range(len(matrix_df.columns)))
ax.set_xticklabels(matrix_df.columns)
ax.set_yticks(range(len(matrix_df.index)))
ax.set_yticklabels(matrix_df.index)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Test confusion matrix (n={matrix.sum()})")

for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        value = matrix[row, col]
        text_color = "white" if value > matrix.max() / 2 else "black"

        ax.text(
            col, row, str(value),
            ha="center",
            va="center",
            color=text_color
        )

fig.colorbar(image, ax=ax, label="Image count")
fig.tight_layout()

metrics_df = pd.read_csv(
    project_dir / "classification_metrics.csv",
    index_col="class"
)

metric_columns = ["precision", "recall", "f1"]
metric_names = ["Precision", "Recall", "F1"]

fake_values = metrics_df.loc["FAKE", metric_columns].to_numpy()
real_values = metrics_df.loc["REAL", metric_columns].to_numpy()


x = np.arange(len(metric_columns))
width = 0.35

fig_metrics, ax_metrics = plt.subplots(figsize=(8, 5))

bars_fake = ax_metrics.bar(
    x - width / 2, fake_values, width, label="FAKE"
)

bars_real = ax_metrics.bar(
    x + width / 2, real_values, width, label="REAL"
)

ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(metric_names)
ax_metrics.set_ylim(0, 1)
ax_metrics.set_ylabel("Score (0–1)")
ax_metrics.set_title("Classification metrics on test set")
ax_metrics.legend()

ax_metrics.bar_label(
    bars_fake,
    labels=[f"{value:.2%}" for value in fake_values],
    padding=3,
    fontsize=8
)

ax_metrics.bar_label(
    bars_real,
    labels=[f"{value:.2%}" for value in real_values],
    padding=3,
    fontsize=8
)

fig_metrics.tight_layout()
fig_metrics.savefig(
    project_dir / "classification_metrics.png",
    dpi=200
)


plt.show()