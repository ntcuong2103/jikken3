"""
#Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes.

Results example: ![Visualization](http://i.imgur.com/4nj4KjN.jpg)
"""
from __future__ import print_function

import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 2,3'
from mnist_cnn import load_mnist

x_train, y_train, x_test, y_test = load_mnist()

prev_filters = np.load('filters_incre_conv2d_4.npz')['filters']
prev_filters = [prev_filters[i][0] for i in range(len(prev_filters))]
prev_filters = np.array(prev_filters)


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.

    # Arguments
        x: A numpy-array representing the generated image.

    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    """utility function to convert a valid uint8 image back into a float array.
       Reverses `deprocess_image`.

    # Arguments
        x: A numpy-array, which could be used in e.g. imshow.
        former: The former numpy-array.
                Need to determine the former mean and variance.

    # Returns
        A processed numpy-array representing the generated image.
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def visualize_layer(model,
                    layer_name,
                    step=0.005,
                    epochs=10,
                    upscaling_steps=1,
                    upscaling_factor=1,
                    output_dim=(28, 28),
                    filter_range=(0, None),
                    max_retry=500,
                    init='train_random',
                    center=True,
                    ):
    """Visualizes the most relevant filters of one conv-layer in a certain model.

    # Arguments
        model: The model containing layer_name.
        layer_name: The name of the layer to be visualized.
                    Has to be a part of model.
        step: step size for gradient ascent.
        epochs: Number of iterations for gradient ascent.
        upscaling_steps: Number of upscaling steps.
                         Starting image is in this case (80, 80).
        upscaling_factor: Factor to which to slowly upgrade
                          the image towards output_dim.
        output_dim: [img_width, img_height] The output image dimensions.
        filter_range: Tupel[lower, upper]
                      Determines the to be computed filter numbers.
                      If the second value is `None`,
                      the last filter will be inferred as the upper boundary.
    """

    def _generate_filter_image(input_img,
                               layer_output,
                               filter_index,
                               num_channels=1,
                               ):
        """Generates image for one particular filter.

        # Arguments
            input_img: The input-image Tensor.
            layer_output: The output-image Tensor.
            filter_index: The to be processed filter number.
                          Assumed to be valid.

        #Returns
            Either None if no image could be generated.
            or a tuple of the image (array) itself and the last loss.
        """
        s_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered

        out_shape = layer_output.get_shape()

        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            if center:
                loss = K.mean(layer_output[:, int((int(out_shape[1]) + 1)/2) - 1, int((int(out_shape[2]) + 1)/2) - 1, filter_index])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some random noise
        intermediate_dim = tuple(
            int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)

        # print(intermediate_dim)

        for retry in range(max_retry):
            skip = False

            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random(
                    (1, num_channels, intermediate_dim[0], intermediate_dim[1]))
            else:
                if init == 'random':
                    input_img_data = np.random.random(
                        (1, intermediate_dim[0], intermediate_dim[1], num_channels))
                elif init == 'filters':
                    input_img_data = np.average(prev_filters[np.random.randint(len(prev_filters), size=(1,))], axis=0)
                    input_img_data = input_img_data / np.max(input_img_data)
                    input_img_data = np.expand_dims(input_img_data, 0)
                elif init == 'train_index':
                    ## choose random image from x_train with filter index
                    indexes = np.where(y_train[:,filter_index] == 1)
                    indexes = indexes[0]
                    input_img_data = x_train[np.random.choice(indexes, size=1)]
                elif init == 'train_random':
                    ## choose random image from x_train
                    input_img_data = x_train[np.random.randint(len(x_train), size=(1,))]

            # Slowly upscaling towards the original size prevents
            # a dominating high-frequency of the to visualized structure
            # as it would occur if we directly compute the 412d-image.
            # Behaves as a better starting point for each following dimension
            # and therefore avoids poor local minima
            for up in reversed(range(upscaling_steps)):
                # we run gradient ascent for e.g. 20 steps
                for _ in range(epochs):
                    loss_value, grads_value = iterate([input_img_data])
                    input_img_data += grads_value * step
                    # some filters get stuck to 0, we can skip them
                    if loss_value <= K.epsilon():
                        skip = True
                        break

                if skip: break

                # Calulate upscaled dimension
                intermediate_dim = tuple(
                    int(x / (upscaling_factor ** up)) for x in output_dim)
                # Upscale
                img = deprocess_image(input_img_data[0])

                import cv2
                img = cv2.resize(img, intermediate_dim)
                img = np.expand_dims(img, -1)

                input_img_data = [process_image(img, input_img_data[0])]

            if not skip:
                print('retries = {}'.format(retry + 1))
                break

        # decode the resulting input image
        img = deprocess_image(input_img_data[0])
        e_time = time.time()
        print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index, loss_value, e_time - s_time))
        return img, loss_value

    def _draw_filters(filters, n=None):
        """Draw the best filters in a nxn grid.

        # Arguments
            filters: A List of generated images and their corresponding losses
                     for each processed filter.
            n: dimension of the grid.
               If none, the largest possible square will be used
        """
        if n is None:
            # n = int(np.floor(np.sqrt(len(filters))))
            n = int(np.ceil(np.sqrt(len(filters))))

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top n*n filters.
        # filters.sort(key=lambda x: x[1], reverse=True)
        # filters = filters[:n * n]

        # build a black picture with enough space for
        # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
        MARGIN = 5
        width = n * output_dim[0] + (n - 1) * MARGIN
        height = n * output_dim[1] + (n - 1) * MARGIN
        stitched_filters = np.zeros((width, height, 3), dtype='uint8')

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                if i*n + j < len(filters):
                    img, _ = filters[i * n + j]
                    width_margin = (output_dim[0] + MARGIN) * i
                    height_margin = (output_dim[1] + MARGIN) * j
                    stitched_filters[
                        width_margin: width_margin + output_dim[0],
                        height_margin: height_margin + output_dim[1], :] = img

        # save the result to disk
        save_img('{0:}_{1:}_{2:}x{2:}.png'.format(model.name, layer_name, n), stitched_filters)

    # this is the placeholder for the input images
    assert len(model.inputs) == 1
    input_img = model.inputs[0]

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    output_layer = layer_dict[layer_name]
    assert isinstance(output_layer, layers.Conv2D)

    print ('weight shape = {}, bias shape = {}'
           .format(output_layer.get_weights()[0].shape, output_layer.get_weights()[1].shape))

    # Compute to be processed filter range
    filter_lower = filter_range[0]
    filter_upper = (filter_range[1]
                    if filter_range[1] is not None
                    else len(output_layer.get_weights()[1]))
    assert(filter_lower >= 0
           and filter_upper <= len(output_layer.get_weights()[1])
           and filter_upper > filter_lower)
    print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))

    # iterate through each filter and generate its corresponding image
    processed_filters = []
    for f in range(filter_lower, filter_upper):
        img_loss = _generate_filter_image(input_img, output_layer.output, f)

        if img_loss is not None:
            processed_filters.append(img_loss)

    print('{} filter processed.'.format(len(processed_filters)))

    # saved filters
    np.savez('filters_incre_%s.npz' %(layer_name), filters=np.array(processed_filters))

    # Finally draw and store the best filters to disk
    _draw_filters(processed_filters)


if __name__ == '__main__':
    pass
