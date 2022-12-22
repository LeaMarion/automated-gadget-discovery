import yaml
from data_mining.pattern import Pattern
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import combinations, product

def generate_hsh(matrix):
    """
    Generates hsh from matrix according to the python hash function. Need to convert the matrix to a tuple first.
    Args:
        matrix (np.array) the current observation in form of a matrix

    Returns:
         hsh (int) the hash of the current observation
    """
    hsh = hash(tuple(matrix.flatten()))
    return hsh

def get_config(config, experiment):
    path = 'configurations/'+experiment+'/'+config
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def default_pattern():
    name = [np.array([0,0,0,0])]
    id = -1
    F = 0.0
    C = 0.0
    I = 0.0
    default_pattern = Pattern(name, id, F, C, I)
    
    return default_pattern


def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0

    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] = [c, x, num_qubits, 0]
        i += 1

    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits),
                        range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary


def dict_of_actions_revert_q(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0

    for c, x in product(range(num_qubits - 1, -1, -1),
                        range(num_qubits - 1, 0, -1)):
        dictionary[i] = [c, x, num_qubits, 0]
        i += 1

    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits - 1, -1, -1),
                        range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary




def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.ylim(0.,1.)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def render_circuit(circuit, reward, NUM_QUDITS, NUM_OPS):
    element_positions = list(combinations(range(NUM_QUDITS), 2))
    width = NUM_OPS * 50 + 100
    height = NUM_QUDITS * 50 + 100
    im = Image.new('RGB', (width, height), (194, 194, 198))
    draw = ImageDraw.Draw(im)
    draw.polygon(((0, 0), (width, height), (width, 0)), fill=(194, 194, 198))
    draw.polygon(((0, 0), (width, height), (0, height)), fill=(194, 194, 198))
    if reward == 1.0:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)
    x_pos = 60 + 50 * NUM_OPS
    y_pos = 10
    draw.rectangle((x_pos, y_pos, x_pos + 30, y_pos + 30), fill=color)

    y_start = 70

    for i in range(NUM_QUDITS):
        draw.line(((0, y_start + 10 + 50 * i), (width, y_start + 10 + 50 * i)), fill=(255, 255, 255), width=5)

    for j, element in enumerate(circuit):
        x_pos = 50 + 50 * j
        if element[0] == 1:
            if j%2 == 0:
                color = (0, 0, 255)
            else:
                color = (0, 0, 255)
                # color = (95, 173, 233)
        else:
            if j%2 == 0:
                color = (255, 0, 0)
            else:
                color = (255, 0, 0)
                # color = (255, 122, 122)

        if element[1] == 1:
            pos_1 = (element[2]-1) * 50
            draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)

        else:
            pos_1, pos_2 = element_positions[element[2] - 1]
            pos_1 = pos_1 * 50
            pos_2 = pos_2 * 50
            draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)
            draw.ellipse((x_pos, y_start + pos_2, 20 + x_pos, y_start + 20 + pos_2), fill=color)
            draw.line(((x_pos + 10, y_start + 10 + pos_1), (x_pos + 10, y_start + 10 + pos_2)), fill=color, width=5)

    im.show()