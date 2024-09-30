import torch
import cv2
import subprocess
import sys
sys.path.insert(0, "/home/pxn-lyj/Egolee/programs/yolov7-face")
from models.experimental import attempt_load
from models.yolo import Model
from tools_utils import GpuMemoryCalculator
from utils.datasets import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from torch2trt import TRTModule
import tensorrt as trt
import numpy as np
from utils.torch_utils import select_device
from tools_utils import PostProcessor



def debug_load_yolo(device="cuda:0", imgsz=640):
    imgs = torch.zeros(1, 3, imgsz, imgsz).to(device)

    gpu_memory_calculator = GpuMemoryCalculator()


    gpu_memory_usage_0 = gpu_memory_calculator.get_device_gpu_memory_usage()

    # model = Model(cfg='/home/pxn-lyj/Egolee/programs/yolov7-face/cfg/yolov7-lite-s.yaml')
    # model = Model(cfg='/home/pxn-lyj/Egolee/programs/yolov7-face/cfg/yolov7s-face.yaml')
    # model.to(device)
    # model.eval()

    weights_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-lite-s.pt"
    model = attempt_load(weights_path, map_location=device)  # load
    model.eval()

    gpu_memory_usage = gpu_memory_calculator.get_device_gpu_memory_usage() - gpu_memory_usage_0
    print("gpu_memory_usage: {}".format(gpu_memory_usage))


def test_yolo_gpu_mem_and_speed(model, source, test_mode, device="cuda:0", imgsz=(768, 1280),
                                batch_size=1, use_half=False, max_count=200, gpu_mem_calculator=None,
                                postprocess=None):

    model.eval()
    dataset = LoadImages(source, img_size=imgsz, stride=32)

    torch.cuda.empty_cache()

    count = 0
    m_allocateds = []
    m_reserveds = []
    m_max_reserveds = []
    all_gpu_usages = []
    gpu_usage_1s = []
    infer_times = []
    post_times = []

    while 1:
        for path, imgs, im0s, vid_cap in dataset:
            # imgs = torch.from_numpy(imgs).to(device, non_blocking=True)
            imgs = torch.from_numpy(imgs).to(device)
            imgs = imgs.half() if use_half else imgs.float()  # uint8 to fp16/32
            imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
            if imgs.ndimension() == 3:
                imgs = imgs.unsqueeze(0)
                imgs = imgs.repeat(batch_size, 1, 1, 1)

            if use_half:
                imgs.half()

            t1 = time_synchronized()
            if test_mode == "torch":
                with torch.no_grad():
                    # pred = model(imgs, augment=False)[0]
                    pred = model(imgs)[0]
            else:
                pred = model(imgs)

            t2 = time_synchronized()

            # nms
            if test_mode == "torch2trt":
                pred = postprocess(list(pred))[0]

            if test_mode == "trt_engine":
                pred = pred[0]

            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None,
                                       agnostic=False, kpt_label=5)

            t3 = time_synchronized()

            # visualize
            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                kpt_label = 5
                if len(det):
                    scale_coords(imgs.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                    scale_coords(imgs.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                    # Write results
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        c = int(cls)  # integer class
                        label = None
                        kpts = det[det_index, 6:]
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                     line_thickness=3, kpt_label=kpt_label, kpts=kpts, steps=3,
                                     orig_shape=im0.shape[:2])

            cv2.imwrite("img_tmp.jpg", im0)

            m_allocated = gpu_mem_calculator.get_torch_memory_allocated() - gpu_mem_calculator.m_allocated_0
            m_reserved = gpu_mem_calculator.get_torch_memory_reserved() - gpu_mem_calculator.m_reserved_0
            m_max_reserved = gpu_mem_calculator.get_torch_max_memory_reserved() - gpu_mem_calculator.m_max_reserved_0

            gpu_usage_1 = gpu_mem_calculator.get_device_gpu_memory_usage()
            all_gpu_usage = gpu_usage_1 - gpu_mem_calculator.gpu_usage_0

            infer_time = (t2 - t1) * 1000
            post_time = (t3 - t2) * 1000

            print("m_allocated:{}, m_reserved:{}, m_max_reserved:{}, all_gpu_usage:{},"
                  " used memory: {}, infer_time: {}ms, post_time: {} ms".format(m_allocated, m_reserved, m_max_reserved,
                                                            all_gpu_usage, gpu_usage_1, infer_time, post_time))

            count = count + 1
            m_allocateds.append(m_allocated)
            m_reserveds.append(m_reserved)
            m_max_reserveds.append(m_max_reserved)
            all_gpu_usages.append(all_gpu_usage)
            gpu_usage_1s.append(gpu_usage_1)
            infer_times.append(infer_time)
            post_times.append(post_time)

        if count > max_count:
            break

    mean_m_allocated = round(np.mean(m_allocateds[20:]), 2)
    mean_m_reserved = round(np.mean(m_reserveds[20:]), 2)
    mean_m_max_reserved = round(np.mean(m_max_reserveds[20:]), 2)
    mean_all_gpu_usage = round(np.mean(all_gpu_usages[20:]), 2)
    mean_gpu_usage_1 = round(np.mean(gpu_usage_1s[20:]), 2)
    mean_infer_time = round(np.mean(infer_times[20:]), 2)
    mean_post_time = round(np.mean(post_times[20:]), 2)

    print("mean_m_allocated: {}M, mean_m_reserved: {}M, mean_m_max_reserved: {}M, mean_all_gpu_usage: {}M, mean_gpu_usage_1: {}M,"
          " mean_infer_time: {}ms, mean_post_time: {} ms".format(mean_m_allocated, mean_m_reserved, mean_m_max_reserved,
                                                                 mean_all_gpu_usage, mean_gpu_usage_1, mean_infer_time, mean_post_time))


if __name__ == "__main__":
    print("Start")
    # 采用.yaml和.pt的方式分别进行模型的加载，发现采用.pt的方式占用显存会更小一些
    # debug_load_yolo()

    # 进行模型显存和速度测评
    device = "cuda:0"
    test_mode = "trt_engine"    # torch, trt_engine, torch2trt
    imgsz = (768, 1280)
    batch_size = 4
    use_fp16 = True
    source = "/home/pxn-lyj/Egolee/programs/yolov7-face_liyj/data/images"
    postprocess = PostProcessor(device)

    imgs = torch.zeros(batch_size, 3, imgsz[0], imgsz[1]).to(device)
    if use_fp16:
        imgs.half()

    # 不包含context-memory的显存占用情况
    gpu_mem_calculator = GpuMemoryCalculator()
    gpu_mem_calculator.gpu_usage_0 = gpu_mem_calculator.get_device_gpu_memory_usage()
    gpu_mem_calculator.m_allocated_0 = gpu_mem_calculator.get_torch_memory_allocated()
    gpu_mem_calculator.m_reserved_0 = gpu_mem_calculator.get_torch_memory_reserved()
    gpu_mem_calculator.m_max_reserved_0 = gpu_mem_calculator.get_torch_max_memory_reserved()

    if test_mode == "torch":
        # torch model
        # weights_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-lite-t.pt"
        # weights_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-lite-s.pt"
        weights_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-tiny.pt"
        # weights_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-w6-face.pt"
        model = attempt_load(weights_path, map_location=device)  # load

    if test_mode == "trt_engine":
        use_fp16 = False    # 采用trt_engine的方式, 如果保存的engine为fp16，但是如果输入为fp16会报错，待查明原因
        # 在启用 FP16 模式时,您需要确保输入数据类型为 FP16 或 FP32,TensorRT 会自动进行数据类型转换

        # trt-engine model
        # load trt model
        # trt_engine_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-lite-s.trt"
        # trt_engine_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-w6-face.trt"
        trt_engine_path = "/home/pxn-lyj/Egolee/programs/yolov7-face_liyj/local_files/yolov7-tiny.trt"
        logger = trt.Logger(trt.Logger.INFO)

        with open(trt_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        # model = TRTModule(engine=engine, input_names=["images"], output_names=['991', '1000', '1009'])
        # model = TRTModule(engine=engine, input_names=["images"],
        #                       output_names=['1964', 'onnx::Slice_1007', 'onnx::Slice_1333', 'onnx::Slice_1658'])  # with grid

        # model = TRTModule(engine=engine, input_names=["images"],
        #                       output_names=['1815', 'onnx::Slice_533', 'onnx::Slice_859', 'onnx::Slice_1184', 'onnx::Slice_1509'])  # with grid

        model = TRTModule(engine=engine, input_names=["images"],
                          output_names=['1312', 'onnx::Slice_355', 'onnx::Slice_681', 'onnx::Slice_1006'])  # with grid

    if test_mode == "torch2trt":
        torch2trt_weight_path = "/home/pxn-lyj/Egolee/programs/yolov7-face_liyj/local_files/yolov7-tiny_trt_batch_4.pth"
        model = TRTModule()
        model.load_state_dict(torch.load(torch2trt_weight_path, map_location=device))

    if use_fp16:
        model.half()
    torch.cuda.empty_cache()

    test_yolo_gpu_mem_and_speed(model, source, test_mode, device=device,
                                imgsz=imgsz, batch_size=batch_size, use_half=use_fp16,
                                max_count=200, gpu_mem_calculator=gpu_mem_calculator,
                                postprocess=postprocess)
    print("End")