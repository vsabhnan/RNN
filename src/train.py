import time
import pandas as pd
from sklearn.metrics import f1_score
import torch, random, numpy as np
import tensorflow as tf

results = []  # stores metrics for all runs
results_csv = "experiment_results.csv"

def train_record_results(model,
                         X_train, y_train,
                         X_test, y_test,
                         run_config,
                         epochs=5, batch_size=32):
    start_time = time.time()

    # Set seed
    tf.random.set_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    end_time = time.time()

    train_time_per_epoch = (end_time - start_time) / epochs

    # Predictions → convert probabilities → binary class
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

    # Compute metrics
    accuracy = (y_pred.flatten() == y_test).mean()
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # Record result row
    result_row = {
        **run_config,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "train_time_per_epoch_sec": train_time_per_epoch
    }

    results.append(result_row)

    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)

    return accuracy, f1_macro, train_time_per_epoch