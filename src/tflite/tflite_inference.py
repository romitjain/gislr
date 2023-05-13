import argparse

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
        required=True
    )

    parser.add_argument(
        '--infer_path',
        type=str,
        required=False
    )

    args = parser.parse_args()

    # Convert the model
    print("Loading tflite model..")

    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(args.model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    print(input_shape)

    if args.infer_path:
        # Input will be (Nframes * 543 * 3)
        # Reshape appropriately
        # Do feature engg
        input_data = np.load(args.infer_path)

        og_size = input_data.shape[0]
        bfill = True if np.random.uniform() < 0.5 else False
        for i in range(0, 32-og_size):
            frame_to_add = i % og_size
            if bfill:
                frame_to_add = abs(og_size-i) % og_size

            input_data = np.concatenate((
                input_data, np.expand_dims(input_data[frame_to_add], axis=0)
            ))

        input_data = np.where(np.isnan(input_data), 0, input_data)

        input_data = np.expand_dims(input_data, axis = 0)
        input_data = np.array(input_data, dtype=np.float32)

        print(input_data.shape)

    else:
        print("Generating random data....")
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = tf.nn.softmax(output_data)
    print(np.argmax(output_data))
