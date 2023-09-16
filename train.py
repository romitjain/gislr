import argparse
import tensorflow as tf

from src.data_generator import GISLRSequence
from src.model import GISLRModel, GISLRModelv2
import src.utils as utils

# Constants
save_dataset_path = "./data/generated/np_landmarks/"
class_path = "./data/raw/train.csv"
class_to_label_path = "./data/raw/sign_to_prediction_index_map.json"
val_participants = [55372, 61333, 62590]
test_participants = [30680, 37779, 2044]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        required=False
    )

    parser.add_argument(
        '--sample_size',
        type=int,
        default=32,
        required=False
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=False
    )

    args = parser.parse_args()

    df = utils.get_train_dataset(
        save_dataset_path=save_dataset_path,
        class_path=class_path,
        class_to_label_path=class_to_label_path
    )

    df = utils.get_example_set(
        df, test_participants=test_participants, val_participants=val_participants
    )

    print(df.groupby("example_set").participant_id.count())

    train_dataset = GISLRSequence(
        df[df.example_set == "train"],
        x_col="save_dataset_path",
        y_col="y_label",
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )

    val_dataset = GISLRSequence(
        df[df.example_set == "val"],
        x_col="save_dataset_path",
        y_col="y_label",
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        example_set="val"
    )

    test_participants = GISLRSequence(
        df[df.example_set == "test"],
        x_col="save_dataset_path",
        y_col="y_label",
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        example_set="test"
    )

    print(f"Dataset size: Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_participants)}")

    for x, _ in train_dataset:
        print(x.shape)
        break

    model = GISLRModelv2(
        input_shape=x.shape[1:],
        n_classes=df.y_label.nunique()
    ).get_model()

    if args.model_path:
        print("Loading model...")
        model = tf.keras.models.load_model(args.model_path)
        print(model.summary())

    history = model.fit(
        train_dataset,
        epochs=100,
        verbose=1,
        validation_data=val_dataset,
        callbacks=utils.get_model_callbacks(epoch_size=len(train_dataset)),
        use_multiprocessing=True,
        workers=4
    )
