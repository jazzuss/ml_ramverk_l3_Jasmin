EXPERIMENTS = [
    {
        "name": "baseline",
        "params": {
            "lr": 1e-3,
            "batch_size": 64,
            "epochs": 10,
            "dropout": 0.25,
        },
    },
    {
        "name": "high_lr",
        "params": {
            "lr": 5e-3,
            "batch_size": 128,
            "epochs": 10,
            "dropout": 0.25,
        },
    },
    {
        "name": "regularized",
        "params": {
            "lr": 1e-3,
            "batch_size": 64,
            "epochs": 15,
            "dropout": 0.40,
        },
    },
]