import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Proportions must sum to 1.0!")

    random.seed(seed)

    train_dir = Path(target_dir) / 'train'
    val_dir = Path(target_dir) / 'val'
    test_dir = Path(target_dir) / 'test'

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    source_dir = Path(source_dir)

    print("Start splitting dataset...")
    for class_folder in tqdm(list(source_dir.iterdir())):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)

        all_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            all_files.extend(list(class_folder.glob(ext)))
            all_files.extend(list(class_folder.glob(ext.upper())))

        random.shuffle(all_files)

        n_files = len(all_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)

        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]

        for file, dest_dir in [
            (train_files, train_dir),
            (val_files, val_dir),
            (test_files, test_dir)
        ]:
            for src_path in file:
                dst_path = dest_dir / class_name / src_path.name
                shutil.copy2(src_path, dst_path)

    print("\nSplit has been finished!")
    print("\nStatistics:")
    for dataset_type, directory in [("train", train_dir),
                                    ("validation", val_dir),
                                    ("test", test_dir)]:
        n_examples = sum(len(list(Path(directory / d).glob('*')))
                         for d in os.listdir(directory))
        print(f"Set {dataset_type}: {n_examples} images")

    print("\nDataset structure summary:")
    for dataset_type, directory in [("train", train_dir),
                                    ("val", val_dir),
                                    ("test", test_dir)]:
        print(f"\n{dataset_type}/")
        for class_dir in sorted(os.listdir(directory)):
            n_files = len(list(Path(directory / class_dir).glob('*')))
            print(f"  └─ {class_dir}/: {n_files} images")


if __name__ == "__main__":
    SOURCE_DIR = Path(__file__).parent / ".." / "dataset" / "raw-img"
    SOURCE_DIR = SOURCE_DIR.resolve()
    TARGET_DIR = Path(__file__).parent / ".." / "data"
    TARGET_DIR = TARGET_DIR.resolve()

    split_dataset(source_dir=SOURCE_DIR, target_dir=TARGET_DIR, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
