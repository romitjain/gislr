import argparse
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
        required=True
    )

    parser.add_argument(
        '--save_path',
        type=str,
        required=False
    )

    args = parser.parse_args()

    # Convert the model
    print("Converting model to tflite model..")
    converter = tf.lite.TFLiteConverter.from_saved_model(args.model_path)
    tflite_model = converter.convert()

    # Save the model
    print("Saving tflite model..")
    save_path = "./model.tflite" if args.save_path is None else args.save_path
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
