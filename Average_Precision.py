import os
from glob import glob
import shutil
import json
import sys
import numpy as np
from m_main import get_configurations

args = get_configurations()

""" composed of 3 parts = file preset, calculation, results"""

"""
1. file preset part
"""

# path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# make sure that the cwd() is the location of the python script (so that every path makes sense)

GT_PATH = os.path.join(os.getcwd(), 'input', 'ground_truth')
DR_PATH = os.path.join(os.getcwd(), 'input', 'detection_results')
IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')

# # image path
# if os.path.exists(IMG_PATH):
#     for dirpath, dirnames, files in os.walk(IMG_PATH):
#         if not files:
#             args.no_animation = True
# else:
#     args.no_animation = True


# make temp path
TEMP_FILES_PATH = "temp_files"
if not os.path.exists(TEMP_FILES_PATH):
    os.makedirs(TEMP_FILES_PATH)


# make results path
result_path = "result_file"
if os.path.exists(result_path):
    shutil.rmtree(result_path)
    os.makedirs(result_path)
elif not os.path.exists(result_path):
    os.makedirs(result_path)

# load all csv lists in ground_truth
ground_truth_files_list = glob(GT_PATH + '/*.txt')
ground_truth_files_list.sort()

# load all detection_result files
dr_files_list = glob(DR_PATH + '/*.txt')
dr_files_list.sort()

""" Convert the rows of a txt!! file to list """
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like '\n' at the end of each line
    content = [x.strip() for x in content]
    return content

'''ignore'''
if args.ignore is None:
    args.ignore = []

"""error msg"""
def error(msg):
    print(msg)
    sys.exit(0)

"""check the range of figure"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


'''load ground truth files and elements
    Load each of the ground-truth files into a temporary ".json" file
'''
def get_gt_lists(GT_PATH, DR_PATH, TEMP_FILES_PATH, args):
    if len(ground_truth_files_list) == 0:
        error("Error: There is no gt file in GT_PATH")
    ground_truth_files_list.sort()

    # count element
    gt_counter_per_class = {}
    counter_images_per_classes = {}

    for txt_file in ground_truth_files_list:
        # example file name = 2007_00027.txt
        file_id = txt_file.split(".txt", 1)[0]  # 2007_00027
        file_id = os.path.basename(os.path.normpath(file_id))  # set as path
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))

        if not os.path.exists(temp_path):
            error_msg = "there is no detection results matched gt file: {}\n".format(temp_path)
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)  # gt file's row --> list

        # create gt dictionary
        bounding_boxes = []
        already_seen_classes = []

        for line in lines_list:
            try:
                class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: file" + txt_file + "in the wrong format.\n"
                error(error_msg)
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

            # count how many gts are in one class(dictionary)
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class did not exits yet
                gt_counter_per_class[class_name] = 1

            # count how many classes in one image(dictionary)
            if class_name not in already_seen_classes:
                if class_name in counter_images_per_classes:
                    counter_images_per_classes[class_name] += 1
                else:
                    # if class did not exist yet
                    counter_images_per_classes[class_name] = 1
                already_seen_classes.append(class_name)

        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)

    return gt_counter_per_class, counter_images_per_classes



"""
 detection-results
     Load each of the detection-results files into a temporary ".json" file.
"""
def load_dr_into_json(GT_PATH, dr_files_list, TEMP_FILE_PATH, gt_classes):
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # the first time it checks if all the corresponding ground truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)
            lines = file_lines_to_list(txt_file)

            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    # match
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)

        with open(TEMP_FILE_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)


"""
check format
"""
def check_format_class_iou(args, gt_classes):
    n_args = len(args.set_class_IoU)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)

    specific_iou_classes = args.set_class_IoU[::2]  # even elements
    iou_list = args.set_class_IoU[1::2]  # odd elements
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            error('Error, unknown class \"' + tmp_class + '\".Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IOU must be between 0 and 1. Flag usage:' + error_msg)


"""
2. calculattion part
"""

"""Overall Calculation Frame"""
def voc_ap(rec, prec):
    rec.insert(0, 0.0)  # insert 0.0 at beginning of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    # This part makes the precision monotonically decreasing as recall increases (end-->beginning)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # This part creates a list of index, where the recall changes (from 1.0 to 0.9 / 0.9 to 0.8 )
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    # The Average Precision (AP) is the average point in precision
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i]) #intergrating version, no need ap_sum, just ap
    return ap, mrec, mpre


"""
calculate fp, tp, prec, rec
"""
def compute_pre_rec(fp, tp, class_name, gt_counter_per_class):
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx] / gt_counter_per_class[class_name])

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx] / (fp[idx] + tp[idx]))

    return rec, prec


"""
interpolation
"""
def calc_interpolated_prec(desired_rec, latest_pre, rec, prec):
    recall_precision = np.array([rec, prec])
    recall_precision = recall_precision.T

    inter_recall = recall_precision[recall_precision[:, 0] >= desired_rec]
    inter_precision = inter_recall[:, 1]

    if len(inter_precision) > 0:
        inter_precision = max(inter_precision)
        latest_pre = inter_precision
    else:
        inter_precision = 0
    return inter_precision, latest_pre


"""
interpolation
"""
def calc_inter_ap(args, rec, prec):
    inter_precisions = []
    latest_pre = 0
    for i in range(args.N_inter):
        recall = float(i) / (args.N_inter - 1)
        inter_precision, latest_pre = calc_interpolated_prec(recall, latest_pre, rec, prec)
        inter_precisions.append(inter_precision)
    return np.array(inter_precisions).mean()


"""get ap and map"""
def calculate_ap(TEMP_FILE_PATH, results_files_path, gt_classes, args,
                 gt_counter_per_class, counter_images_per_class):
    specific_iou_flagged = False
    if args.set_class_IoU is not None:
        specific_iou_flagged = True

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the results
    with open(results_files_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class \n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0

            '''load detection results of that class'''
            dr_file = TEMP_FILE_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            '''Assign detection results to gt objects'''
            nd = len(dr_data)
            tp = [0] * nd  #make zero array, size = nd
            fp = [0] * nd

            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                # assign detection results to gt object if any
                # open gt with that file id
                gt_file = TEMP_FILE_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                IoUmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [float(x) for x in detection["bbox"].split()]
                confidence = float(detection["confidence"])
                # if confidence < opt.confidence_threshold:
                #    fp[idx] = 1
                #    continue
                for obj in ground_truth_data:
                    # look for class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        # 순서: left top right bottom
                        # bi = detection과 gt 중 교집합 box의 좌표
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # ua = compute overlap (IoU) = area of intersection/ area of union
                            ua = ((bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1)
                                  * (bbgt[3] - bbgt[1] + 1)) - iw*ih
                            IoU = iw * ih / ua
                            if IoU > IoUmax:
                                IoUmax = IoU
                                gt_match = obj

                # assign detection as true positive/false positive
                # set minimum overlap threshold
                IoU_th = args.IoU_th
                if specific_iou_flagged:
                    specific_iou_classes = args.set_class_IoU[::2]
                    iou_list = args.set_class_IoU[1::2]
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        IoU_th = float(iou_list[index])
                if IoUmax >= IoU_th:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update json file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))

                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1

                else:
                    fp[idx] = 1

            rec, prec = compute_pre_rec(fp, tp, class_name, gt_counter_per_class)

            if args.no_interpolation:
                ap, mrec, mprec = voc_ap(rec[:], prec[:])
            else:
                ap = calc_inter_ap(args, rec[:], prec[:])
            # ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = class_name + " AP " + " = " + "{0:.2f}%".format(ap * 100) # class_name + " AP = {0:.2f}%".format(ap*100)
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if not args.quiet:
                print(text)
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            # lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            # lamr_dictionary[class_name] = lamr

        results_file.write("\n-----mAP of all classes-----\n")
        mAP = sum_AP / len(gt_classes)
        text = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)
    return count_true_positives


"""3. making results part"""


gt_counter_per_class, counter_images_per_class = get_gt_lists(GT_PATH, DR_PATH, TEMP_FILES_PATH, args)

gt_classes = list(gt_counter_per_class.keys())
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

load_dr_into_json(GT_PATH, dr_files_list, TEMP_FILES_PATH, gt_classes)




'''Count total of detection-results'''
det_counter_per_classes = {}
for txt_file in dr_files_list:
    # get lines to list
    lines_list = file_lines_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        # check if class is in the ignore list, if yes skip
        if class_name in args.ignore:
            continue
        if class_name in det_counter_per_classes:
            det_counter_per_classes[class_name] += 1
        else:
            # class did not exist yet
            det_counter_per_classes[class_name] = 1

dr_classes = list(det_counter_per_classes.keys())


if args.set_class_IoU is not None:
    check_format_class_iou(gt_classes)




# """Plot - adjust axes"""
# def adjust_axes(r, t, fig, axes):
#     # get text width for re-scaling
#     bb = t.get_window_extent(rendered=r)
#     text_width_inches = bb.width / fig.dpi
#     # get axis width in inches
#     current_fig_width = fig.get_figwidth()
#     new_fig_width = current_fig_width + text_width_inches
#     propotion = new_fig_width / current_fig_width
#
#     # get axis limit
#     x_lim = axes.get_xlim()
#     axes.set_xlim([x_lim[0], x_lim[1]*propotion])



'''Write num of gt object per classes to results.txt'''
with open(result_path + "/results.txt", 'a') as results_file:
    results_file.write("\n----- Number of gt objects per class-----\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ":" + str(gt_counter_per_class[class_name]) + "\n")

count_true_positives = calculate_ap(TEMP_FILES_PATH, result_path, gt_classes, args,
                                    gt_counter_per_class, counter_images_per_class)


'''Finish counting tp'''
for class_name in dr_classes:
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0

with open(result_path + "/results.txt", 'a') as results_file:
    results_file.write("\n----- Number of detected objects per class-----\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_classes[class_name]
        text = class_name + ": " + str(n_det) #+ "\n"
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        results_file.write(text)



shutil.rmtree(TEMP_FILES_PATH)


