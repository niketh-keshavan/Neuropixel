from pathlib import Path
import numpy as np
import readSGLX


def convert_string_to_tuples(data_str):
    """
    Convert a formatted string to a list of tuples with integers.

    :param data_str: String in the format '(2013,384)(0 0 0 0 0)(1 0 0 0 1)...'
    :return: List of tuples containing integers.
    """
    # Remove the surrounding parentheses and split by ')('
    data_str = data_str.strip('()')
    entries = data_str.split(')(')

    # Convert each entry to a tuple of integers
    tuples_list = []
    for entry in entries:
        parts = entry.split(' ')
        tuple_entry = tuple(map(int, parts))
        tuples_list.append(tuple_entry)

    return tuples_list


def create_channel_map(channel_info):
    """
    Create a 4x48x2 matrix mapping channels to their physical positions.

    :param channel_info: List of tuples (channel_number, shank_number, 0, 0, position_on_shank).
    :return: 4x48x2 numpy array representing the channel map.
    """
    # Initialize the 4x48x2 matrix with -1 (indicating empty positions)
    channel_map = np.full((4, 48, 2), -1, dtype=int)

    for info in channel_info:
        channel_number = info[0]
        shank_number = info[1]
        position_on_shank = info[4]

        # Calculate row and column indices in the shank's 2D matrix
        row = 48 - position_on_shank // 2 - 1
        column = position_on_shank % 2

        # Place the channel number in the appropriate position
        channel_map[shank_number, row, column] = channel_number

    return channel_map


def calculate_xy_positions(channel_map):
    """
    Calculate the x and y positions of the center of each electrode given its index and the channel map.

    :param channel_map: 3D numpy array representing the channel map with dimensions [shank, row, column].
    :return: Dictionary with channel numbers as keys and (x, y) positions as values.
    """
    shank_spacing = 250  # Separation between shanks in um
    electrode_width = 12  # Electrode width in um
    electrode_height = 12  # Electrode height in um
    horizontal_spacing = 20  # Horizontal spacing between electrodes on the same shank in um
    vertical_spacing = 3  # Vertical spacing between electrodes on the same shank in um

    positions = {}

    # Iterate through the channel map to calculate positions
    for shank in range(channel_map.shape[0]):
        for row in range(channel_map.shape[1]):
            for col in range(channel_map.shape[2]):
                channel_number = channel_map[shank, row, col]
                if channel_number != -1:  # Only calculate for valid channels
                    # Calculate x position
                    x = shank * shank_spacing + col * (electrode_width + horizontal_spacing)
                    # Calculate y position
                    y = (47-row) * (electrode_height + vertical_spacing)
                    # Adjust for the center of the electrode
                    x_center = x + electrode_width / 2
                    y_center = y + electrode_height / 2
                    positions[channel_number] = (x_center, y_center)
    sorted_positions = sorted(positions.items())

    return sorted_positions


bin_path = Path('C:\\Users\\niket\\Documents\\Neuropixel\\Heart_1\\240610_Heart1\\240610_Heart1_start0_end10.exported.imec0.ap.bin')
meta = readSGLX.readMeta(bin_path)
channel_info = convert_string_to_tuples(meta['imroTbl'][10:])
channel_map = create_channel_map(channel_info)
xy_position = calculate_xy_positions(channel_map)
print('Electrode positions: ', xy_position)
