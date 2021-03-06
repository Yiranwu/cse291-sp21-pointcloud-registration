import numpy as np
import imageio

CLASS_NAMES = ['a_cups', 'a_lego_duplo', 'a_toy_airplane', 'adjustable_wrench', 'b_cups', 'b_lego_duplo',
               'b_toy_airplane', 'banana', 'bleach_cleanser', 'bowl', 'bowl_a', 'c_cups', 'c_lego_duplo',
               'c_toy_airplane', 'cracker_box', 'cup_small', 'd_cups', 'd_lego_duplo', 'd_toy_airplane', 'e_cups',
               'e_lego_duplo', 'e_toy_airplane', 'extra_large_clamp', 'f_cups', 'f_lego_duplo', 'flat_screwdriver',
               'foam_brick', 'fork', 'g_cups', 'g_lego_duplo', 'gelatin_box', 'h_cups', 'hammer', 'i_cups', 'j_cups',
               'jenga', 'knife', 'large_clamp', 'large_marker', 'master_chef_can', 'medium_clamp', 'mug',
               'mustard_bottle', 'nine_hole_peg_test', 'pan_tefal', 'phillips_screwdriver', 'pitcher_base', 'plate',
               'potted_meat_can', 'power_drill', 'prism', 'pudding_box', 'rubiks_cube', 'scissors', 'spoon',
               'sugar_box', 'tomato_soup_can', 'tuna_fish_can', 'wood_block', 'pigeon', 'pure_zhen', 'realsense_box',
               'pepsi', 'green_arrow', 'red_car', 'conditioner', 'correction_fuid', 'wooden_puzzle1', 'wooden_puzzle2',
               'green_car', 'redbull', 'doraemon_cup', 'doraemon_bowl', 'hello_kitty_cup', 'hello_kitty_bowl',
               'hello_kitty_plate', 'doraemon_spoon', 'tea_can1', 'wooden_puzzle3']
# len=79
EVAL_CLASS_IDS = None


class Evaluator(object):
    def __init__(self, class_names=CLASS_NAMES, labels=None):
        self.class_names = tuple(class_names)
        self.num_classes = len(class_names)
        self.labels = (
            np.arange(self.num_classes) if labels is None else np.array(labels)
        )
        assert self.labels.shape[0] == self.num_classes
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.nb_samples = np.zeros(self.num_classes)

    def update(self, y_pred, y_true):
        num_classes = self.num_classes

        mask = (y_true != num_classes)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        # intersection
        intersection = y_pred[(y_pred == y_true)]
        intersection = np.histogram(intersection, bins=num_classes, range=[0, num_classes - 1])[0]

        # union
        total_num_pred = np.histogram(y_pred, bins=num_classes, range=[0, num_classes - 1])[0]
        total_num_label = np.histogram(y_true, bins=num_classes, range=[0, num_classes - 1])[0]
        union = total_num_pred + total_num_label - intersection

        self.intersection += intersection
        self.union += union
        self.nb_samples += total_num_label

    @property
    def overall_seg_acc(self):
        return np.sum(self.intersection) / np.sum(self.nb_samples)

    @property
    def overall_iou(self):
        return np.nanmean(self.intersection / self.union)

    @property
    def class_seg_acc(self):
        return [self.intersection[i] / self.nb_samples[i]
                for i in range(self.num_classes)]

    @property
    def class_iou(self):
        iou_list = self.intersection / self.union
        return iou_list

    def print_table(self):
        from tabulate import tabulate

        header = ["Class", "SegAccuracy", "IOU", "Total"]
        seg_acc_per_class = self.class_seg_acc
        iou_per_class = self.class_iou
        table = []
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          seg_acc_per_class[ind] * 100,
                          iou_per_class[ind] * 100,
                          int(self.nb_samples[ind]),
                          ])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate

        header = ("overall acc", "overall iou") + self.class_names
        table = [[self.overall_seg_acc, self.overall_iou] + self.class_iou]
        with open(filename, "w") as f:
            # In order to unify format, remove all the alignments.
            f.write(
                tabulate(
                    table,
                    headers=header,
                    tablefmt="tsv",
                    floatfmt=".5f",
                    numalign=None,
                    stralign=None,
                )
            )


def merge(evaluator1, evaluator2):
    new_eval = Evaluator()
    new_eval.union = evaluator1.union + evaluator2.union
    new_eval.intersection = evaluator1.intersection + evaluator2.intersection
    return new_eval


def main():
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate IoU")
    parser.add_argument(
        "--pred-dir", type=str, help="path to prediction",
    )
    parser.add_argument(
        "--gt-dir", type=str, help="path to ground-truth",
    )
    args = parser.parse_args()

    gt_files = [f for f in os.listdir(args.gt_dir) if f.endswith(".png")]
    pred_files = []
    for i in range(len(gt_files)):
        pred_file = os.path.join(args.pred_dir, gt_files[i])
        if not os.path.isfile(pred_file):
            raise RuntimeError("No matched prediction for {}.".format(gt_files[i]))
        pred_files.append(pred_file)
        gt_files[i] = os.path.join(args.gt_dir, gt_files[i])

    evaluator = Evaluator(CLASS_NAMES, EVAL_CLASS_IDS)
    print("Evaluating", len(gt_files), "samples...")

    # sync
    for i in range(len(pred_files)):
        # It takes a long time to load data.
        pred_label = imageio.imread(pred_files[i])
        gt_label = imageio.imread(gt_files[i])
        evaluator.update(pred_label, gt_label)
        sys.stdout.write("\rsamples processed: {}".format(i + 1))
        sys.stdout.flush()

    print("")
    print(evaluator.print_table())


if __name__ == "__main__":
    main()
