import numpy as np
from utils.configs import CONFIGS

def judge_rubbish_is_normal(pred_pw_area, pred_npw_area):
        normal_flag = True
        # 厨余垃圾和非厨余垃圾像素全为0
        if 0 == pred_pw_area and 0 == pred_npw_area:
                normal_flag = False
        # 厨余垃圾像素为0， 非厨余垃圾像素和<1000
        elif 0 == pred_pw_area and pred_npw_area < CONFIGS.pix_sum_thre:
                normal_flag = False
        # 非厨余垃圾像素为0， 厨余垃圾像素和<1000
        elif 0 == pred_npw_area and pred_pw_area < CONFIGS.pix_sum_thre:
                normal_flag = False

        return normal_flag


# 三个类别： 1和2和3
def calc_diff_three_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count):
        result_rate = 0.0
        GT_rate = 0.0

        # 计算pt或rknn预测精度
        # 把非0的都置1， 并计算所有像素和
        pred_all = pred.copy()
        pred_all[pred_all == 2] = 1
        pred_all[pred_all == 3] = 1
        pred_all_area = np.sum(pred_all)  # 统计厨余垃圾和非厨余垃圾的像素和

        pred_pw = pred.copy()
        pred_pw[pred_pw == 2] = 0  # 非厨余垃圾像素置于0，剩下的就只有厨余垃圾的值为1
        pred_pw[pred_pw == 3] = 0
        pred_pw_area = np.sum(pred_pw)  # 统计全部非厨余垃圾的值的像素和

        pred_npw_area = pred_all_area - pred_pw_area
        print("所有类别像素和{}, 厨余垃圾像素和{}, 非厨余垃圾像素和{}".format(pred_all_area, pred_pw_area, pred_npw_area))
        normal_flag = judge_rubbish_is_normal(pred_pw_area, pred_npw_area)
        if True == normal_flag:
                result_rate = 0.0 if 0 == pred_all_area else pred_pw_area / pred_all_area
                print("厨余垃圾类别像素和", pred_pw_area)
                #result_rate_list.append(result_rate)

                # 计算gt的精度
                GT_all = mask_np.copy()
                GT_all[GT_all == 2] = 1
                GT_all[GT_all == 3] = 1
                GT_all_area = np.sum(GT_all)
                GT_pw = mask_np.copy()
                GT_pw[GT_pw == 2] = 0
                GT_pw[GT_pw == 3] = 0
                GT_pw_area = np.sum(GT_pw)
                GT_rate = 0.0 if 0 == GT_all_area else GT_pw_area / GT_all_area
                #GT_rate_list.append(GT_rate)

                total_count = total_count + 1

                # if abs(GT_rate - result_rate) < 0.1:
                #     normal_count = normal_count + 1
                # if abs(GT_rate - result_rate) < 0.15:
                #     no_bad_count = no_bad_count + 1
        else:
                print("<<图像有问题， 可能是空桶，可能是盖着桶>>")

        return normal_count, no_bad_count, total_count, result_rate, GT_rate, normal_flag


# 两个类别： 1和2
def calc_diff_two_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count):
        result_rate = 0.0
        GT_rate = 0.0

        # 计算pt或rknn预测精度
        pred_all = pred.copy()
        pred_all[pred_all == 2] = 1
        pred_all_area = np.sum(pred_all)  # 统计厨余垃圾和非厨余垃圾的像素和

        pred_pw = pred.copy()
        pred_pw[pred_pw == 2] = 0  # 厨余垃圾像素置于0，剩下的就只有非厨余垃圾的值为1
        pred_pw_area = np.sum(pred_pw)  # 统计全部非厨余垃圾的值的像素和

        pred_npw_area = pred_all_area - pred_pw_area
        print("所有类别像素和{}, 厨余垃圾像素和{}, 非厨余垃圾像素和{}".format(pred_all_area, pred_pw_area, pred_npw_area))
        normal_flag = judge_rubbish_is_normal(pred_pw_area, pred_npw_area)
        if True == normal_flag:
                result_rate = 0.0 if 0 == pred_all_area else pred_pw_area / pred_all_area
                print("厨余垃圾类别像素和", pred_pw_area)
                # 计算gt的精度
                GT_all = mask_np.copy()
                GT_all[GT_all == 2] = 1
                GT_all_area = np.sum(GT_all)
                GT_pw = mask_np.copy()
                GT_pw[GT_pw == 2] = 0
                GT_pw_area = np.sum(GT_pw)
                GT_rate = 0.0 if 0 == GT_all_area else GT_pw_area / GT_all_area
                #GT_rate_list.append(GT_rate)

                total_count = total_count + 1

                # if abs(GT_rate - result_rate) < 0.1:
                #     normal_count = normal_count + 1
                # if abs(GT_rate - result_rate) < 0.15:
                #     no_bad_count = no_bad_count + 1
        else:
                print("<<图像有问题， 可能是无桶，可能是盖着盖子>>")
        return normal_count, no_bad_count, total_count, result_rate, GT_rate, normal_flag

# 1个类别： 1
def calc_diff_one_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count, self_test=0):
        # 计算pt或rknn预测精度
        pred_all = pred.copy()
        pred_all[pred_all != 0] = 1
        pred_all_area = np.sum(pred_all) # 统计厨余垃圾和非厨余垃圾的像素和
        # pred_all_area = sum(map(sum, pred_all)) # 统计厨余垃圾和非厨余垃圾的像素和

        # 计算gt的精度
        GT_all = mask_np.copy()
        GT_all[GT_all != 0] = 1
        GT_all_area = np.sum(GT_all)
        # GT_all_area = sum(map(sum, GT_all))

        # 测试1
        if 0 == self_test:
            error_rate = abs(int(pred_all_area) - int(GT_all_area)) / GT_all_area if 0 != GT_all_area else 0.0
        # 测试2
        elif 1 == self_test:
            if 0 != GT_all_area:
                error_rate = abs(pred_all_area - GT_all_area) / GT_all_area
            elif 0 == GT_all_area and 0 == pred_all_area:
                error_rate = 0.0
            elif 0 == GT_all_area and 0 != pred_all_area:
                error_rate = 1.0

        total_count = total_count + 1

        return normal_count, no_bad_count, total_count, error_rate


