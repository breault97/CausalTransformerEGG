import hydra
from omegaconf import DictConfig, OmegaConf

from runnables.train_multi import run as run_ct
from runnables.train_baselines import _run_baseline


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    # Route to CT training or baseline pipeline
    if hasattr(args.model, "baseline"):
        backbone_name = str(args.model.name)
        _run_baseline(args, backbone_name)
    else:
        run_ct(args)


if __name__ == "__main__":
    main()
