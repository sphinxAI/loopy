import os
import argparse

import Average_Precision as AP
import confidence_threshold as CT


def get_configurations():

    p=argparse.ArgumentParser()

    p.add_argument('--N_inter', type=int, help='number of interpolations', default=11)
    p.add_argument('--IoU_th', type=int, help='IoU threshold', default=50)
    p.add_argument('--C_th', type=int, help='confidence threshold', default=30)
    p.add_argument('--size_threshold', type=int, help='image size(small, medium, large)', default=32)
    p.add_argument('--set_class_IoU', type=int, help='apply to specific class', default=None)
    # p.add_argument('--sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
    p.add_argument('--np', '--noplot', dest='showPlot', action='store_false', help='no plot is shown during execution')
    p.add_argument('--data_path', type=str, help = 'data path', default=os.path.join('.', 'input'))
    p.add_argument('--ignore', nargs='+', type=str, help="ignore a list of classes.")
    p.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    p.add_argument('no_interpolation', help="interpolation no? yes?", defaullt=False)

    return p.parse_args()



if __name__ == "__main__":
    config = get_configurations()
    # os.chdir(os.path.dirname(os.path.abspath(r'C:\\Users\\Medical-Information\\PycharmProjects\\project_metric\\input')))
    # GT_PATH = os.path.join(os.getcwd(), 'input', 'ground_truth')
    # DR_PATH = os.path.join(os.getcwd(), 'input', 'detection_results')

    CT.CT(config)







