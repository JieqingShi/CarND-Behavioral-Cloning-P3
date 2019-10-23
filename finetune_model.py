# Script for finetuning an already trained model
import argparse
import keras
import numpy as np
from keras.models import load_model
from model import get_train_val_samples, generator, batch_size

def finetune_trained_model(model_path, img_path, learning_rate=0.0001, epochs=5, model_name="model_finetuned.h5"):
    loaded_model = load_model(model_path)

    # freeze fc layers (assuming it's the NVIDIA model, therefore hardcoding the layer numbers)
    for layer in loaded_model.layers[:-5]:
        layer.trainable=False

    train_samples, validation_samples = get_train_val_samples(driving_log_path="{}/driving_log.csv".format(img_path))

    # compile and train the model using the generator function
    train_generator = generator(train_samples, path_img="{}/IMG/".format(img_path), batch_size=batch_size)
    validation_generator = generator(validation_samples, path_img="{}/IMG/".format(img_path), batch_size=batch_size)

    loaded_model.compile(loss=keras.losses.mse,
                         optimizer=keras.optimizers.Adam(lr=learning_rate))

    # model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=5, shuffle=True)
    loaded_model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                               validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), 
                               epochs=epochs, verbose=1)

    loaded_model.save(model_name)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model finetuning')
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to model h5 file. Required with the filename in the form /path/file.h5'
    )
    parser.add_argument(
        '--img_path',
        type=str,
        nargs='?',
        help='Path to new images folder. Only directory required in the form /path/'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate for Adam Optimizer. Set to a factor of 10 smaller than default learning rate in Keras (i.e. 0.0001)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs to finetune the model. Defaults to 5',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='model_finetuned.h5',
        help='The name of the model to be saved. Defaults to model_finetuned.h5'
    )
    args = parser.parse_args()
    
    finetune_trained_model(model_path=args.model_path, img_path=args.img_path, learning_rate=args.learning_rate, epochs=args.epochs, 
                           model_name=args.model_name)
    print("model finetuned for {} epochs with learning rate = {} and saved under {}".format(args.epochs, args.learning_rate, args.model_name))

