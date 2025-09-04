
import argparse
import os
import json
import skimage.io



def load_image(filename):

    return skimage.io.imread(filename)


def save_image(filename, image):

    skimage.io.imsave(filename, image)

    return None


def crop(image, indexes):

    return image[indexes]

def format_range(rang, max_stop):

    if isinstance(rang, int):
        return [0, rang, 1]

    elif isinstance(rang, (list, tuple)):
        for number_i in rang:
            if not isinstance(number_i, int):
                raise TypeError("Each element of rang must be an integer")

    n_numbers = len(rang)
    if n_numbers == 0:

        if not isinstance(max_stop, int):
            raise TypeError("max_stop must be an integer")

        return [0, max_stop, 1]

    elif n_numbers == 1:
        return [0, rang[0], 1]

    elif n_numbers == 2:
        return [rang[0], rang[1], 1]

    elif n_numbers == 3:
        return [rang[0], rang[1], rang[2]]
    else:
        raise ValueError('Too many numbers in rang')


def format_ranges(ranges, maxes_stops):

    n_ranges = len(ranges)
    n_maxes_stops = len(maxes_stops)

    if n_ranges != n_maxes_stops:
        raise ValueError('ranges and maxes_stops must have the same number of elements')

    for i in range(0, n_ranges, 1):

        ranges[i] = format_range(rang=ranges[i], max_stop=maxes_stops[i])

    return ranges


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog=None, usage=None, description=None, epilog=None, parents=[],
        formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None,
        argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True,
        exit_on_error=True)

    parser.add_argument(
        'input_image_filename', action='store', type=str,
        help='The filename of the input image to crop.')

    parser.add_argument(
        '-o', '--output_image_filename', action='store', default=None, type=str, required=False,
        help='The filename of the output cropped image. Default is <input_image_filename_no_extension>_crop<input_image_file_extension>.')

    parser.add_argument(
        '-s', '--output_range_filename', action='store', default=None, type=str, required=False,
        help='The JSON filename with the index ranges used to crop the image. Default is <output_image_filename_no_extension>_range.json')

    parser.add_argument(
        '-r', '--ranges', action='store', default=None, type=str, required=False,
        help='The JSON string with the index ranges of image to crop. For instance, the string "[[100, 200], [300, 500, 2]]" will crop the image with:\n`image = image[tuple([slice(100, 200, 1), slice(300, 500, 2)])]`\nInstead, the the string "[[300], []]" will crop the image with:\n`image = image[tuple([slice(0, 300, 1), slice(0, image.shape[1], 1)])]`\n')

    parser.add_argument(
        '-c', '--dim_channel', action='store', default=None, type=int, required=False,
        help='The channel dimension of the Image to crop. If None (default), the image is considered to have no channel dimensions.)')

    args = parser.parse_args()

    if args.output_image_filename is None:
        dirname, basename = os.path.split(args.input_image_filename)
        basename_no_ext, ext = os.path.splitext(basename)
        args.output_image_filename = os.path.join(basename_no_ext + '_crop' + ext)

    os.makedirs(os.path.dirname(args.output_image_filename), exist_ok=True)

    if args.output_range_filename is None:
        output_image_filename_no_ext, ext = os.path.splitext(args.output_image_filename)
        args.output_range_filename = os.path.join(output_image_filename_no_ext + '_range.json')
    else:
        os.makedirs(os.path.dirname(args.output_range_filename), exist_ok=True)

    image = load_image(filename=args.input_image_filename)

    if args.ranges is None:

        ranges = []
    else:
        ranges = json.loads(args.ranges)

    if len(ranges) > 0:

        ranges = format_ranges(ranges=ranges, maxes_stops=image.shape)

        slices = [slice(*range_i) for range_i in ranges]

        if args.dim_channel is not None:
            slice_channel = slice(0, image.shape[args.dim_channel], 1)
            slices.insert(args.dim_channel, slice_channel)

        image = crop(image=image, indexes=tuple(slices))

    elif args.dim_channel is None:
        ranges = [[0, shape_a, 1] for shape_a in image.shape]
    else:
        ranges = [[0, image.shape[a], 1] for a in range(0, image.ndim, 1) if a != args.dim_channel]



    with open(args.output_range_filename, "w") as text_file:
        text_file.write(json.dumps(ranges, indent=2))

    save_image(filename=args.output_image_filename, image=image)
