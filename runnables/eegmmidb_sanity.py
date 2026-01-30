import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    """
    Quick sanity check: load a small eegmmidb subset and print feature dimensions.
    Use CLI overrides to limit records, e.g. dataset.subjects=1 dataset.runs=1 dataset.max_seq_length=5.
    """
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    train = dataset_collection.train_f
    val = dataset_collection.val_f
    test = dataset_collection.test_f

    logger.info(f"Train features: {train.data['current_covariates'].shape}")
    logger.info(f"Val features:   {val.data['current_covariates'].shape}")
    logger.info(f"Test features:  {test.data['current_covariates'].shape}")
    logger.info(f"Feature dim (F): {train.data['current_covariates'].shape[-1]}")
    logger.info(f"Sequence length (T): {train.data['current_covariates'].shape[1]}")
    logger.info("Sanity check complete.")


if __name__ == "__main__":
    main()
