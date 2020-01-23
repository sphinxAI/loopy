import os
import argparse

import Average_Precision as AP
import confidence_threshold as CT


def get_configurations():
    parser = argparse.ArgumentParser()

    parser.add_argument('--N_inter', type=int, help='number of interpolations', default=11)
    parser.add_argument('--IoU_th', type=int, help='IoU threshold', default=0.50)
    parser.add_argument('--C_th', type=int, help='confidence threshold', default=30)
    parser.add_argument('--set_class_IoU', type=int, help='apply to specific class', default=None)
    parser.add_argument('--ignore', nargs='+', type=str, help="ignore a list of classes.", default=None)
    parser.add_argument('--quiet', help="minimalistic console output.", action="store_true", default=True)
    parser.add_argument('--no_interpolation', help="interpolation no? yes?", default=False)

    parser.add_argument('--na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('--npl', '--noplot', dest='showPlot', action='store_false',
                        help='no plot is shown during execution')
    parser.add_argument('--data_path', type=str, help='data path', default=os.path.join('.', 'input'))
    # parser.add_argument('--size_threshold', type=int, help='image size(small, medium, large)', default=32)

    args = parser.parse_args()

    return args


if __name__ == "__m_main__":

    configuration = get_configurations()
    # os.chdir(os.path.dirname(os.path.abspath(r'C:\\Users\\Medical-Information\\PycharmProjects\\project_metric\\input')))
    # GT_PATH = os.path.join(os.getcwd(), 'input', 'ground_truth')
    # DR_PATH = os.path.join(os.getcwd(), 'input', 'detection_results')

    AP.AP(configuration)
