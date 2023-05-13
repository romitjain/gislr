import os
import json
import pandas as pd
import tensorflow as tf
from datetime import datetime


def get_example_set(
        x: pd.DataFrame,
        val_participants: list,
        test_participants: list
        ) -> pd.DataFrame:

    def get_example_set_row_wise(x):
        if x.participant_id in test_participants:
            return "test"
        elif x.participant_id in val_participants:
            return "val"
        return "train"

    x["example_set"] = x.apply(
        get_example_set_row_wise, axis = 1
    )

    return x

def get_train_dataset(
        save_dataset_path: str, class_path: str, class_to_label_path: str
    ) -> pd.DataFrame:
    class_name = pd.read_csv(class_path)
    print(f"Schema: {class_name.columns}")

    with open(class_to_label_path, "r") as fp:
        y_label = json.load(fp)
        y_label = pd.DataFrame.from_dict(y_label, orient="index", columns = ["y_label"])

    df = class_name.set_index("sign").join(y_label).reset_index()

    df["save_dataset_path"] = df.sequence_id.apply(lambda x: os.path.join(save_dataset_path + str(x) + ".npy"))

    return df


def get_model_callbacks(epoch_size: int):
    callbacks = []

    # Tensorboard logging
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join("./logs/fit/", datetime.now().strftime("%Y%m%d-%H%M")),
        histogram_freq=1
    )
    callbacks.append(tensorboard_callback)

    # Model checkpointing
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(f'./model/{datetime.now().strftime("%Y%m%d-%H%M")}', 'cp-{epoch:02d}'),
        save_freq="epoch",
        monitor="val_loss",
        save_best_only=True,
        period=8
    )
    callbacks.append(checkpoint)

    # Learning rate scheduler
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        verbose=1,
        mode='auto',
        min_delta=0.01,
        cooldown=0,
        min_lr=1e-8,
    )
    callbacks.append(lr_schedule)

    # Custom learning rate scheduler
    # custom_lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lr_schedule,
    #     verbose=1
    # )
    # callbacks.append(custom_lr_schedule)

    return callbacks


def lr_schedule(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 10


def get_all_files(path):
    all_files = []

    for dir, _, fls in os.walk(path):
        for fl in fls:
            all_files.append(os.path.join(dir, fl))

    print(f"Found: {len(all_files)} files")

    return all_files
