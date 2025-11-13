from models import build_model
from train import train_record_results

def run_iter(run_index, arch, activation, opt, grad, seq_length, datasets):

    data = datasets[str(seq_length)]
    X_tr, y_tr = data["train"]
    X_te, y_te = data["test"]

    config = {
        "run": run_index,
        "architecture": arch,
        "activation": activation,
        "optimizer": opt,
        "sequence_length": seq_length,
        "gradient_clipping": grad
    }

    model = build_model(
        architecture=arch,
        activation=activation,
        optimizer=opt,
        clip=grad,
        sequence_length=seq_length
    )

    acc, f1, t, history = train_record_results(
        model,
        X_tr, y_tr,
        X_te, y_te,
        run_config=config,
        epochs=5
    )

    print(f"{arch}: Acc={acc:.4f}, F1={f1:.4f}, Time/Epoch={t:.2f}s")
    return history