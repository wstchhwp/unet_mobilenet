
class CONFIGS(object):

    RKNN_RUN_BY_COMPUTER_CPU = 1
    RKNN_RUN_BY_COMPUTER_NPU = 0
    RKNN_RUN_BY_RV1126_NPU = 0

    # 模型训练输入的图像大小是多大，这里设置就多大
    pix_sum_thre = 1000  # 像素和阈值
    model_channel = 3
    model_width = 512
    model_height = 512

    img_channel = 3
    img_width = 256
    img_height = 256
    n_class = 4

    epoch = 1799
    checkpoints_folder = "./checkpoints/20220727_syx/"
    checkpoints_pth = "checkpoint_epoch_" + str(epoch) + ".pth"
    checkpoints_pt = "checkpoint_epoch_" + str(epoch) + ".pt"
    checkpoints_pt_jit = "checkpoint_epoch_" + str(epoch) + "_jit.pt"

    # gt: imgs和masks
    ##########################################
    #test_images_path = "/home/bcd/Documents/01lianyun/05test_dataSets/02lugong_renGongPoDai/6mmhuanQiao_150/imgs/"
    #test_masks_path = "/home/bcd/Documents/01lianyun/05test_dataSets/02lugong_renGongPoDai/6mmhuanQiao_150/masks/"
    test_images_path = "/home/bcd/Documents/01lianyun/02AiEdgeCalc/01NPU/01RV1126/06Unet_mobileNet/06py_rk_unet_segmentation/tmp/fuyang/imgs"
    test_masks_path = "/home/bcd/Documents/01lianyun/02AiEdgeCalc/01NPU/01RV1126/06Unet_mobileNet/06py_rk_unet_segmentation/tmp/fuyang/masks/"
    ##########################################


    # pth-->pt
    ##########################################
    pth_path = checkpoints_folder + checkpoints_pth
    pt_jit_path = checkpoints_folder + checkpoints_pt_jit
    # pth_jit_path = './checkpoints/checkpoint_epoch_2000_jit.pth'
    ##########################################


    # pt-->pt_jit
    ##########################################
    pt_path = checkpoints_folder + checkpoints_pt
    pt_jit_path = checkpoints_folder + checkpoints_pt_jit
    ##########################################


    # 030jitpt_to_rknn.py
    # 031rknn_run_inference_pred_gt_cal_diff_value.py
    ##########################################
    input_jit_pt = checkpoints_folder + checkpoints_pt_jit
    model_size_list = [[model_channel, model_width, model_height]]  #rknn.load_pytorch()需要输入的数据维度
    quantization_on = True
    quantization_dataset ='./dataset.txt'
    export_rknn_path = checkpoints_folder + 'unet' + str(epoch) + '.rknn'
    pre_compile = True

    load_rknn_path = export_rknn_path
    rknn_pred_mask_save_folder = "./py_rknn_predMask/"
    ##########################################


    # 040jitPt_Unet_Mnet_gt_pred_cal_diff_value.py
    ##########################################
    pt_pred_mask_save_folder = "./py_pt_predMask/"
    pt_gt_mask_save_folder = "./py_pt_gtMask/"
    ##########################################


    # 042 overflow_diff_value
    ##########################################
    overflow_classes = 2  # 满溢模型的类别数，目前满溢模型只有rubbish和bg两个类别
    # gt: imgs和masks
    # test_overflow_images_path = "/home/bcd/Documents/01lianyun/02AiEdgeCalc/01NPU/01RV1126/06Unet_mobileNet/06py_rk_unet_segmentation/gt_imgs_masks/02overflow_masks/imgs"
    # test_overflow_masks_path = "/home/bcd/Documents/01lianyun/02AiEdgeCalc/01NPU/01RV1126/06Unet_mobileNet/06py_rk_unet_segmentation/gt_imgs_masks/02overflow_masks/masks"
    test_overflow_images_path = "/home/bcd/Documents/01lianyun/05test_dataSets/01overflow_test_dataSets/test_dataSets/imgs"
    test_overflow_masks_path = "/home/bcd/Documents/01lianyun/05test_dataSets/01overflow_test_dataSets/test_dataSets/masks/"

    ##########################################

