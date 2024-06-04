from pathlib import Path
import shutil
import re

from sklearn.model_selection import train_test_split
from loguru import logger
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def data_split(
    random_seed,
    test_split,
    data_dir,
    train_data_dir,
    test_data_dir,
):

    # Create the directories if they don't exist
    train_data_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Data split: Discovering images and masks.")
    # Find the maximum index of images and ground truth masks
    images = list(data_dir.glob("**/image_*.npy"))
    images.sort()
    num_images = len(images)
    masks = list(data_dir.glob("**/mask_*.npy"))
    masks.sort()
    num_masks = len(masks)
    if num_images != num_masks:
        raise ValueError("Different number of images and masks.")

    # Train test split
    image_indexes = [int(re.search(r"\d+", file.name).group()) for file in data_dir.glob("**/image_*.npy")]
    mask_indexes = [int(re.search(r"\d+", file.name).group()) for file in data_dir.glob("**/mask_*.npy")]

    if set(image_indexes) != set(mask_indexes):
        raise ValueError(f"Different image and mask indexes : {image_indexes} and {mask_indexes}")

    logger.info(f"Data split: Found {len(image_indexes)} images and masks in {data_dir}.")
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(images, masks, test_size=test_split, random_state=random_seed)
    logger.info(f"Data split: Split into {len(train_imgs)} training images and {len(test_imgs)} test images.")

    # Copy the images and masks to the train and test directories
    logger.info("Data split: Copying images and masks to train and test directories.")
    for img, msk in zip(train_imgs, train_masks):  # training data
        shutil.copy(img, train_data_dir / img.name)
        shutil.copy(msk, train_data_dir / msk.name)
        # copy over png's too
        shutil.copy(img.parent / str(img.stem + '.png'), train_data_dir / str(img.stem + '.png'))
        shutil.copy(msk.parent / str(msk.stem + '.png'), train_data_dir / str(msk.stem + '.png'))
    
    for img, msk in zip(test_imgs, test_masks): # testing data
        shutil.copy(img, test_data_dir / img.name)
        shutil.copy(msk, test_data_dir / msk.name)
        # copy over png's too
        shutil.copy(img.parent / str(img.stem + '.png'), test_data_dir / str(img.stem + '.png'))
        shutil.copy(msk.parent / str(msk.stem + '.png'), test_data_dir / str(msk.stem + '.png'))

    logger.info(f"Data split: Complete. Saved to {train_data_dir} and {test_data_dir} directories.")


if __name__ == "__main__":
    logger.info("Data split: Loading parameters from params.yaml config file.")
    # Get the parameters from the params.yaml config file
    with open(Path("./params.yaml"), "r") as file:
        all_params = yaml.load(file)
        base_params = all_params["base"]
        data_split_params = all_params["data_split"]

    logger.info("Data split: Converting the paths to Path objects.")
    # Convert the paths to Path objects
    data_dir = Path(data_split_params["data_dir"])
    train_data_dir = Path(data_split_params["train_data_dir"])
    test_data_dir = Path(data_split_params["test_data_dir"])

    # Split the data
    data_split(
        random_seed=base_params["random_seed"],
        test_split=data_split_params["test_split"],
        data_dir=data_dir,
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
    )
