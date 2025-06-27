
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
        'input_filename', action='store', type=str,
        help='The filename of the input image to crop.')

    parser.add_argument(
        '-o', '--output_filename', action='store', default=None, type=str, required=False,
        help='The filename of the output cropped image. Default is crop_<input_filename>')

    parser.add_argument(
        '-r', '--ranges', action='store', type=str, required=False,
        help='The JSON list of index ranges of image to crop.')

    parser.add_argument(
        '-c', '--dim_channel', action='store', default=None, type=int, required=False,
        help='The channel dimension of the Image to crop. If None (default), the image is considered to have no channel dimensions.)')

    args = parser.parse_args()

    if args.output_filename is None:
        dirname, basename = os.path.split(args.input_filename)
        basename_no_ext, ext = os.path.splitext(basename)
        args.output_filename = os.path.join(basename_no_ext + '_crop' + ext)

    image = load_image(filename=args.input_filename)

    print(args.dim_channel)

    if args.ranges is not None:

        args.ranges = json.loads(args.ranges)

        if len(args.ranges) > 0:

            if args.dim_channel is not None:
                range_channel = [0, image.shape[args.dim_channel], 1]
                args.ranges.insert(args.dim_channel, range_channel)

            args.ranges = format_ranges(ranges=args.ranges, maxes_stops=image.shape)

            slices = tuple([slice(*range_i) for range_i in args.ranges])


            print(slices)

            image = crop(image=image, indexes=slices)
    
    save_image(filename=args.output_filename, image=image)

