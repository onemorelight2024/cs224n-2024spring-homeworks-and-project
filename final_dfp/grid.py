import itertools, os, subprocess

# 仅搜索这三个
epses = [0.05, 0.10, 0.15]
coeffs = [0.5, 1.0, 1.5]
proj_dims = [128, 256, 384]

# 固定参数（可按需改）
epochs = 15
batch_size = 8
lr = "1e-5"
mode = "full-model"
strategy = "gradient_surgery"
dropout = 0.3
seed = 11711

os.makedirs("logs", exist_ok=True)

grid = itertools.product(epses, coeffs, proj_dims)

for eps, coeff, proj in grid:
    run_id = f"eps_{eps}_coeff_{coeff}_proj_{proj}"
    log = os.path.join("logs", f"{run_id}.log")
    cmd = [
        "python", "multitask_classifier.py",
        "--use_gpu",
        "--fine-tune-mode", mode,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--hidden_dropout_prob", str(dropout),
        "--siar_coeff", str(coeff),
        "--siar_eps", str(eps),
        "--task_proj_dim", str(proj),
        "--loss_strategy", strategy,
        "--seed", str(seed),
    ]
    print("Running", run_id)
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)