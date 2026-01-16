import os

import cv2
import pandas as pd
from matplotlib import pyplot as plt


def load_training_results(results_path):
    """Load and clean training results CSV file."""
    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()
    return df


def print_training_metrics(results_df, model_name):
    """Print training summary statistics."""
    print(f"{model_name} TRAINING METRICS")
    print(f"Total epochs: {len(results_df)}")
    print(f"Final metrics:")
    print(f"  Box Loss: {results_df['train/box_loss'].iloc[-1]:.4f}")
    print(f"  Class Loss: {results_df['train/cls_loss'].iloc[-1]:.4f}")
    print(f"  DFL Loss: {results_df['train/dfl_loss'].iloc[-1]:.4f}")
    print(f"  Val mAP50: {results_df['metrics/mAP50(B)'].iloc[-1]:.4f}")
    print(f"  Val mAP50-95: {results_df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")


def plot_training_metrics(results_df, model_name):
    """Create 6-panel training visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{model_name} Training Metrics", fontsize=16, fontweight="bold")

    # 1. Training Losses
    ax = axes[0, 0]
    ax.plot(
        results_df["epoch"], results_df["train/box_loss"], label="Box Loss", linewidth=2
    )
    ax.plot(
        results_df["epoch"],
        results_df["train/cls_loss"],
        label="Class Loss",
        linewidth=2,
    )
    ax.plot(
        results_df["epoch"], results_df["train/dfl_loss"], label="DFL Loss", linewidth=2
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("Training Losses", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation mAP
    ax = axes[0, 1]
    ax.plot(
        results_df["epoch"],
        results_df["metrics/mAP50(B)"],
        label="mAP50",
        color="green",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.plot(
        results_df["epoch"],
        results_df["metrics/mAP50-95(B)"],
        label="mAP50-95",
        color="blue",
        linewidth=2,
        marker="s",
        markersize=3,
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("mAP", fontweight="bold")
    ax.set_title("Validation mAP Metrics", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Precision and Recall
    ax = axes[0, 2]
    ax.plot(
        results_df["epoch"],
        results_df["metrics/precision(B)"],
        label="Precision",
        color="orange",
        linewidth=2,
        marker="d",
        markersize=3,
    )
    ax.plot(
        results_df["epoch"],
        results_df["metrics/recall(B)"],
        label="Recall",
        color="purple",
        linewidth=2,
        marker="^",
        markersize=3,
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Precision & Recall", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Learning Rate
    ax = axes[1, 0]
    lr_cols = [col for col in results_df.columns if "lr/pg" in col]
    for i, lr_col in enumerate(lr_cols):
        ax.plot(
            results_df["epoch"], results_df[lr_col], label=f"LR Group {i}", linewidth=2
        )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Learning Rate", fontweight="bold")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Combined Loss
    ax = axes[1, 1]
    total_loss = (
        results_df["train/box_loss"]
        + results_df["train/cls_loss"]
        + results_df["train/dfl_loss"]
    )
    ax.plot(
        results_df["epoch"], total_loss, color="red", linewidth=2, label="Total Loss"
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Total Loss", fontweight="bold")
    ax.set_title("Combined Training Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. F1 Score
    ax = axes[1, 2]
    precision = results_df["metrics/precision(B)"]
    recall = results_df["metrics/recall(B)"]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    ax.plot(
        results_df["epoch"],
        f1,
        color="teal",
        linewidth=2,
        marker="*",
        markersize=4,
        label="F1 Score",
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("F1 Score", fontweight="bold")
    ax.set_title("F1 Score Progression", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return total_loss


def compare_training_curves(baseline_df, attention_df):
    """Plot side-by-side comparison of training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Training Comparison: Baseline vs Attention-Enhanced Model",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Box Loss Comparison
    ax = axes[0, 0]
    ax.plot(
        baseline_df["epoch"],
        baseline_df["train/box_loss"],
        label="Baseline",
        linewidth=2.5,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        attention_df["epoch"],
        attention_df["train/box_loss"],
        label="Attention",
        linewidth=2.5,
        alpha=0.8,
        color="red",
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Box Loss", fontweight="bold")
    ax.set_title("Box Loss Comparison", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Class Loss Comparison
    ax = axes[0, 1]
    ax.plot(
        baseline_df["epoch"],
        baseline_df["train/cls_loss"],
        label="Baseline",
        linewidth=2.5,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        attention_df["epoch"],
        attention_df["train/cls_loss"],
        label="Attention",
        linewidth=2.5,
        alpha=0.8,
        color="red",
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Class Loss", fontweight="bold")
    ax.set_title("Class Loss Comparison", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. mAP50 Comparison
    ax = axes[1, 0]
    ax.plot(
        baseline_df["epoch"],
        baseline_df["metrics/mAP50(B)"],
        label="Baseline",
        linewidth=2.5,
        alpha=0.8,
        color="blue",
        marker="o",
        markersize=4,
        markevery=2,
    )
    ax.plot(
        attention_df["epoch"],
        attention_df["metrics/mAP50(B)"],
        label="Attention",
        linewidth=2.5,
        alpha=0.8,
        color="red",
        marker="s",
        markersize=4,
        markevery=2,
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("mAP50", fontweight="bold")
    ax.set_title("Validation mAP50 Comparison", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Total Loss Comparison
    ax = axes[1, 1]
    baseline_total = (
        baseline_df["train/box_loss"]
        + baseline_df["train/cls_loss"]
        + baseline_df["train/dfl_loss"]
    )
    attention_total = (
        attention_df["train/box_loss"]
        + attention_df["train/cls_loss"]
        + attention_df["train/dfl_loss"]
    )
    ax.plot(
        baseline_df["epoch"],
        baseline_total,
        label="Baseline",
        linewidth=2.5,
        alpha=0.8,
        color="blue",
    )
    ax.plot(
        attention_df["epoch"],
        attention_total,
        label="Attention",
        linewidth=2.5,
        alpha=0.8,
        color="red",
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Total Loss", fontweight="bold")
    ax.set_title("Total Training Loss Comparison", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_metrics_lists(
    metrics_obj, metrics_names=["mAP50", "mAP50-95", "Precision", "Recall", "F1"]
):
    """Extract metric values from metrics object."""
    f1_score = (
        2
        * (metrics_obj.box.mp * metrics_obj.box.mr)
        / (metrics_obj.box.mp + metrics_obj.box.mr + 1e-6)
    )
    return [
        metrics_obj.box.map50,
        metrics_obj.box.map,
        metrics_obj.box.mp,
        metrics_obj.box.mr,
        f1_score,
    ]


def plot_prediction_comparison(
    model1,
    model2,
    img_paths,
    model1_name="Baseline",
    model2_name="Attention",
    max_images=6,
):
    """Plot side-by-side predictions from two models."""
    img_paths = img_paths[:max_images]
    fig, axes = plt.subplots(len(img_paths), 2, figsize=(16, 4 * len(img_paths)))
    if len(img_paths) == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(img_paths):
        # Model 1 predictions
        results1 = model1.predict(img_path, conf=0.25, verbose=False)
        annotated1 = results1[0].plot()
        annotated1_rgb = cv2.cvtColor(annotated1, cv2.COLOR_BGR2RGB)

        # Model 2 predictions
        results2 = model2.predict(img_path, conf=0.25, verbose=False)
        annotated2 = results2[0].plot()
        annotated2_rgb = cv2.cvtColor(annotated2, cv2.COLOR_BGR2RGB)

        # Display
        axes[idx, 0].imshow(annotated1_rgb)
        axes[idx, 0].set_title(
            f"{model1_name} - {os.path.basename(img_path)}\nDetections: {len(results1[0].boxes)}",
            fontsize=10,
            fontweight="bold",
        )
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(annotated2_rgb)
        axes[idx, 1].set_title(
            f"{model2_name} - {os.path.basename(img_path)}\nDetections: {len(results2[0].boxes)}",
            fontsize=10,
            fontweight="bold",
        )
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.show()
