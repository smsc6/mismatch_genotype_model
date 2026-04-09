import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


FEATURE_COLS = [
    "depth",
    "n_match",
    "n_mismatch",
    "match_rate",
    "mismatch_rate",
    "A_to_C", "A_to_G", "A_to_T",
    "C_to_A", "C_to_G", "C_to_T",
    "G_to_A", "G_to_C", "G_to_T",
    "T_to_A", "T_to_C", "T_to_G",
]

N_FEATURES = len(FEATURE_COLS)
N_CLASSES = 3


############### PATHS / OUTPUTS ###############

def get_project_root():
    """
    Resolve project root assuming this script lives in src/
    """
    return Path(__file__).resolve().parents[1]


def make_run_dir(model_name, exp_name):
    """
    Create one output directory for one model run
    """
    out_root = get_project_root() / "outputs" / "model_training_results"
    out_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{model_name}_{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, timestamp


############### PARSING ###############

def parse_training_row(row):
    """
    Convert one training csv row into numeric features and label

    :param row: dict from training csv
    :return: parsed dict or None if bad row
    """
    try:
        x = [float(row[col]) for col in FEATURE_COLS]
        gt = row["gt"]
        y = int(row["y"])
    except (KeyError, ValueError, TypeError):
        return None

    if y not in (0, 1, 2):
        return None

    return {
        "x": x,
        "gt": gt,
        "y": y,
    }


class TrainingCSVIterableDataset(IterableDataset):
    """
    Stream all rows from one csv
    """

    def __init__(self, in_csv):
        super().__init__()
        self.in_csv = in_csv

    def __iter__(self):
        with open(self.in_csv, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                parsed = parse_training_row(row)
                if parsed is None:
                    continue

                yield (
                    torch.tensor(parsed["x"], dtype=torch.float32),
                    torch.tensor(parsed["y"], dtype=torch.long),
                )


############### STREAMING STATS ###############

def compute_train_stats(train_csv):
    """
    One streaming pass over full training csv to compute:
    - feature mean
    - feature std
    - class counts
    - number of rows
    """
    n = 0
    mean = np.zeros(N_FEATURES, dtype=np.float64)
    m2 = np.zeros(N_FEATURES, dtype=np.float64)
    class_counts = np.zeros(N_CLASSES, dtype=np.int64)
    n_bad_rows = 0

    with open(train_csv, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            parsed = parse_training_row(row)
            if parsed is None:
                n_bad_rows += 1
                continue

            x = np.array(parsed["x"], dtype=np.float64)
            y = parsed["y"]

            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            m2 += delta * delta2

            class_counts[y] += 1

    if n == 0:
        raise ValueError("No valid training rows found")

    if n > 1:
        var = m2 / (n - 1)
    else:
        var = np.ones(N_FEATURES, dtype=np.float64)

    std = np.sqrt(var)
    std[std == 0] = 1.0

    return {
        "n_train_rows": int(n),
        "n_bad_train_rows": int(n_bad_rows),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "class_counts": class_counts,
    }

def make_class_weights(class_counts):
    """
    Inverse-frequency class weights for CrossEntropyLoss - ALTER TO CHANGE WEIGHTING
    """
    counts = np.array(class_counts, dtype=np.float64)

    if np.any(counts == 0):
        raise ValueError(
            f"At least one class has zero training examples: {counts.tolist()}"
        )

    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)

############### MODEL ###############

class MLPClassifier(nn.Module):
    """
    Small feed-forward neural net, single hidden layer
    """

    def __init__(self, in_features, hidden_dim, n_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


######## DEVICE/ NORMALIZATION #######

def get_device():
    """
    Choose best device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def normalize_batch(x, mean, std, device):
    """
    Standardize one  minibatch
    """
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    return (x - mean_t) / std_t


################################# TRAINING ##############################

def make_dataloader(in_csv, batch_size):
    """
    Build one dataloader for one csv
    """
    dataset = TrainingCSVIterableDataset(in_csv=in_csv)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def train_one_epoch(model, dataloader, optimizer, criterion, mean, std, device):
    """
    Train for one epoch on full training csv
    """
    model.train()

    total_loss = 0.0
    n_examples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        x_batch = normalize_batch(x_batch, mean, std, device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        n_examples += batch_size

    if n_examples == 0:
        raise ValueError("No training examples seen in this epoch")

    return total_loss / n_examples


################ EVALUATION/ PREDICTIONS ##############

def classification_report_from_confusion(confusion):
    """
    Compute precision/ recall/ f1 from confusion matrix
    """
    report = {}
    total = confusion.sum()

    for cls in range(confusion.shape[0]):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        support = confusion[cls, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        report[str(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": int(support),
        }

    accuracy = float(np.trace(confusion) / total) if total > 0 else 0.0
    report["accuracy"] = accuracy

    macro_precision = np.mean([report[str(i)]["precision"] for i in range(N_CLASSES)])
    macro_recall = np.mean([report[str(i)]["recall"] for i in range(N_CLASSES)])
    macro_f1 = np.mean([report[str(i)]["f1-score"] for i in range(N_CLASSES)])

    weights = np.array([report[str(i)]["support"] for i in range(N_CLASSES)], dtype=np.float64)
    
    if weights.sum() > 0:
        weighted_precision = np.average(
            [report[str(i)]["precision"] for i in range(N_CLASSES)],
            weights=weights,
        )
        weighted_recall = np.average(
            [report[str(i)]["recall"] for i in range(N_CLASSES)],
            weights=weights,
        )
        weighted_f1 = np.average(
            [report[str(i)]["f1-score"] for i in range(N_CLASSES)],
            weights=weights,
        )

    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0


    report["macro avg"] = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1-score": float(macro_f1),
        "support": int(total),
    }

    report["weighted avg"] = {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1-score": float(weighted_f1),
        "support": int(total),
    }

    return report


def evaluate_and_write_predictions(model, test_csv, out_csv, mean, std, device,):
    """
    Stream over full test csv, write per-row predictions, and accumulate metrics
    """
    model.eval()

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    n_examples = 0
    n_correct = 0
    n_bad_rows = 0

    fieldnames = (
        ["row_idx", "chrom", "pos", "ref", "gt", "y_true",
         "y_pred", "prob_0", "prob_1", "prob_2"]
        + FEATURE_COLS
    )

    with open(test_csv, "r") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for row_idx, row in enumerate(reader, start=1):
                parsed = parse_training_row(row)
                if parsed is None:
                    n_bad_rows += 1
                    continue

                x = torch.tensor([parsed["x"]], dtype=torch.float32, device=device)
                y_true = parsed["y"]
                gt = parsed["gt"]

                x = normalize_batch(x, mean, std, device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                y_pred = int(torch.argmax(logits, dim=1).item())

                confusion[y_true, y_pred] += 1
                n_examples += 1
                if y_pred == y_true:
                    n_correct += 1

                out_row = {
                    "row_idx": row_idx,
                    "chrom": row["chrom"],
                    "pos": row["pos"],
                    "ref": row["ref"],
                    "gt": gt,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "prob_0": float(probs[0]),
                    "prob_1": float(probs[1]),
                    "prob_2": float(probs[2]),
                }

                for col in FEATURE_COLS:
                    out_row[col] = row[col]

                writer.writerow(out_row)

    if n_examples == 0:
        raise ValueError("No valid test examples found")

    accuracy = n_correct / n_examples

    return {
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "n_test_examples": n_examples,
        "n_bad_test_rows": n_bad_rows,
        "predictions_csv": str(out_csv),
    }


############### SAVING ###############

def save_model_weights(model, run_dir, model_name, train_stats, config):
    """
    Save PyTorch model weights, normalization stats
    """
    out_path = run_dir / f"{model_name}_weights.pt"

    payload = {
        "model_state_dict": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "train_mean": train_stats["mean"],
        "train_std": train_stats["std"],
        "class_counts": train_stats["class_counts"],
        "config": config,
    }

    torch.save(payload, out_path)
    print(f"Saved model weights to {out_path}")
    return out_path


def save_model_results(eval_results, model_name, exp_name, train_csv, test_csv, train_stats, config, run_dir, timestamp):
    """
    Save one run to JSON
    """
    out_path = run_dir / f"{model_name}_metrics.json"

    confusion = eval_results["confusion_matrix"]
    report = classification_report_from_confusion(confusion)

    results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "exp_name": exp_name,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "config": config,
        "n_train_rows": int(train_stats["n_train_rows"]),
        "n_bad_train_rows": int(train_stats["n_bad_train_rows"]),
        "class_counts_train": train_stats["class_counts"].tolist(),
        "n_test_examples": int(eval_results["n_test_examples"]),
        "n_bad_test_rows": int(eval_results["n_bad_test_rows"]),
        "accuracy": float(eval_results["accuracy"]),
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
        "predictions_csv": eval_results["predictions_csv"],
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {out_path}")
    return out_path


##################### RUN MODEL #####################

def train_single_model(
    model,
    model_name,
    train_csv,
    test_csv,
    exp_name,
    train_stats,
    batch_size=1024,
    epochs=15,
    lr=1e-3,
    weight_decay=0.0,
    save=True,
):
    """
    Train and evaluate one pytorch model
    """
    device = get_device()
    model = model.to(device)

    ## WEIGHTING
    class_weights = make_class_weights(train_stats["class_counts"]).to(device)
    print("Class Weights: ", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    print(f"\nTraining {model_name} on device: {device}")
    print(f"Train rows: {train_stats['n_train_rows']}")
    print(f"Bad train rows skipped: {train_stats['n_bad_train_rows']}")
    print(f"Train class counts: {train_stats['class_counts']}")

    for epoch in range(1, epochs + 1):
        train_loader = make_dataloader(
            in_csv=train_csv,
            batch_size=batch_size,
        )

        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            mean=train_stats["mean"],
            std=train_stats["std"],
            device=device,
        )

        print(f"{model_name} epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")

    run_dir = None
    timestamp = None

    if save:
        run_dir, timestamp = make_run_dir(model_name, exp_name)
        print(f"Run directory: {run_dir}")

    if save:
        predictions_csv = run_dir / f"{model_name}_test_predictions.csv"
    else:
        predictions_csv = get_project_root() / "outputs" / f"{model_name}_temp_test_predictions.csv"

    eval_results = evaluate_and_write_predictions(
        model=model,
        test_csv=test_csv,
        out_csv=predictions_csv,
        mean=train_stats["mean"],
        std=train_stats["std"],
        device=device,
    )

    print(f"\n{model_name} confusion matrix:")
    print(eval_results["confusion_matrix"])
    print(f"{model_name} accuracy: {eval_results['accuracy']:.6f}")
    print(f"{model_name} bad test rows skipped: {eval_results['n_bad_test_rows']}")

    if save:
        config = {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "device": str(device),
            "feature_cols": FEATURE_COLS,
        }

        save_model_weights(
            model=model,
            run_dir=run_dir,
            model_name=model_name,
            train_stats=train_stats,
            config=config,
        )

        save_model_results(
            eval_results=eval_results,
            model_name=model_name,
            exp_name=exp_name,
            train_csv=train_csv,
            test_csv=test_csv,
            train_stats=train_stats,
            config=config,
            run_dir=run_dir,
            timestamp=timestamp,
        )

    return model, eval_results


############### PIPELINE #################

def train_baseline_model(
    train_csv,
    test_csv,
    exp_name,
    save=True,
    batch_size=1024,
    epochs_mlp=15,
    lr_mlp=1e-3,
    hidden_dim=64,
    dropout=0.1,
    weight_decay=1e-5,
    seed=42,
):
    """
    Train MLP on full chr10 csv and evaluate on full chr11 csv
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_stats = compute_train_stats(train_csv=train_csv)

    print(f"Computed train stats from {train_stats['n_train_rows']} training rows")

    mlp_model = MLPClassifier(
        in_features=N_FEATURES,
        hidden_dim=hidden_dim,
        n_classes=N_CLASSES,
        dropout=dropout,
    )

    train_single_model(
        model=mlp_model,
        model_name="pytorch_mlp",
        train_csv=train_csv,
        test_csv=test_csv,
        exp_name=exp_name,
        train_stats=train_stats,
        batch_size=batch_size,
        epochs=epochs_mlp,
        lr=lr_mlp,
        weight_decay=weight_decay,
        save=save,
    )


################## TESTING ##################
if __name__ == "__main__":
    TRAIN_CSV = "/projects/willerslev/people/vmk372/anzick_project/outputs/chr10_training_data_v1.csv"
    TEST_CSV = "/projects/willerslev/people/vmk372/anzick_project/outputs/chr11_training_data_v1.csv"

    EXP_NAME = "weighted_chr10_train_chr11_test"

    train_baseline_model(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        exp_name=EXP_NAME,
        save=True,
        batch_size=1024,
        epochs_mlp=15,
        lr_mlp=1e-3,
        hidden_dim=64,
        dropout=0.1,
        weight_decay=1e-5,
        seed=42,
    )