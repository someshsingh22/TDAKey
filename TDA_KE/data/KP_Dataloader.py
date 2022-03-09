from pathlib import Path

from flair.datasets import ColumnCorpus

from TDA_KE.data import dataset_folders as folders


def get_dataloader(
    dataset: str,
    data_path: str = "./data",
    columns: dict = {0: "text", 1: "BIO"},
    downsample_train: float = 0.0,
    train: bool = True,
    test: bool = False,
    dev: bool = False,
    tag_type: str = "seq_label",
    verbose=False,
):

    DATA_ROOT = Path(data_path)
    DATASET_PATH = DATA_ROOT / folders[dataset]
    TRAIN_FILE = "train.txt" if train else None
    DEV_FILE = "dev.txt" if dev else None
    TEST_FILE = "test.txt" if test else None

    corpus = ColumnCorpus(
        DATASET_PATH,
        columns,
        train_file=TRAIN_FILE,
        test_file=TEST_FILE,
        dev_file=DEV_FILE,
    )
    if downsample_train > 0:
        corpus.downsample(percentage=downsample_train, only_downsample_train=True)
    stats = corpus.obtain_statistics()
    if verbose:
        print(stats)
    return corpus
