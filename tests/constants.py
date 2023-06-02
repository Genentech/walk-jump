CONFIG_PATH = "../src/walkjump/hydra_config"
TRAINER_OVERRIDES = [
    "++trainer.accelerator=cpu",
    "++data.csv_data_path=data/poas.csv.gz",
    "++dryrun=true",
]
