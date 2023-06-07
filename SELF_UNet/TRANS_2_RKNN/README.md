[2022.7.28 v1.1.2.0]
1. 主要修改了易腐垃圾的占比的代码.

    1.1 pt和rknn的run_inference_pred_gt_cal_diff_value()新增计算误差在5%的代码 if abs(GT_rate - result_rate) < 0.05: good_count = good_count + 1. 
    
2. pt转量化pt，再转rknn. 

    2.1 新增pt转量化pt的接口代码，013pt_to_quantized_pt.py （测试OK）

    2.2 新增032quantionjitpt_to_rknn.py，量化的jitpt转rknn还未成功。

    2.3 修改了unet模型代码，Unet_Mnet_Quant.py，在需要量化的代码位置，改为量化方式。

3. 新增了满溢检测的逻辑判断和满溢检测的http server代码。

    3.1 新增判断满溢逻辑代码042jitPt_Unet_Mnet_gt_pred_cal_overflow_diff_value.py

    3.2 http server代码050http_server_unet_overflow.py 作为server端，由postman作为client.

4. 其他

    4.1 新增calc_diff.py代码中的接口：
    calc_diff_three_label_between_pred_gt() 用于计算背景+其他3个类别的易腐垃圾的占比。
    calc_diff_two_label_between_pred_gt()用于计算背景+其他2个类别的易腐垃圾的占比。
    calc_diff_one_label_between_pred_gt()用于计算背景+其他1个类别的垃圾的满溢率。

    4.2 030jitpt_to_rknn.py的rknn.config中一定要去掉output_optimize=1， force_builtin_perm=False两个参数。


[2022.6.29 v1.1.1.0]
1. 优化了整个代码结构，并且使参数配置都放在utils/config下面。
2. 拆分了之前的代码，拆为030jitpt_to_rknn.py和031rknn_run_inference_pred_gt_cal_diff_value.py两个部分代码。
3. 如何运行，查看RUN_README.md文档。


[2022.5.31 v1.1.0.1]

0. 该版本查看git网址： 
http://192.168.10.70/aicloud/rubbishDetect/03py_rk_unet_segmentation/-/tree/master/01Unet_pt_rknn
由于上述版本的代码结构不好，重新创建了git地址，单独存放py_rk_unet_seg项目。
1. rknn的unet关于python版本代码。
2. 有转换模型代码和模型推理代码。
3. 优化了画图的部分代码。
4. 如何运行，查看RUN_README.md文档。
