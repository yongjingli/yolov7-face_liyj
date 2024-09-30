import numpy as np
import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
import tensorrt as trt


def save_onnx2trt_engine(onnx_model_path=None, trt_engine_path=None, fp16_mode=False):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    """
    仅适用TensorRT V8版本
    生成cudaEngine，并保存引擎文件(仅支持固定输入尺度)
    Serialized engines不是跨平台和tensorRT版本的
    fp16_mode: True则fp16预测
    onnx_model_path: 将加载的onnx权重路径
    trt_engine_path: trt引擎文件保存路径
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    # config.max_workspace_size = GiB(1)
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_model_path, 'rb') as model:
        assert parser.parse(model.read())
        serialized_engine = builder.build_serialized_network(network, config)

    with open(trt_engine_path, 'wb') as f:
        f.write(serialized_engine)  # 序列化

    print('TensorRT file in ' + trt_engine_path)


def debug_save_trt_engine(onnx_model_path, trt_engine_path, fp16_mode=False):
    # onnx_model_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-lite-s.onnx"
    # onnx_model_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-w6-face.onnx"
    # onnx_model_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-tiny.onnx"
    # trt_engine_path = onnx_model_path.replace(".onnx", ".trt")

    save_onnx2trt_engine(onnx_model_path, trt_engine_path, fp16_mode=fp16_mode)


def load_trt_engine(trt_engine_path, device="cuda:0"):
    # trt_engine_path = "/home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-lite-s.trt"
    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    print("1:", config.get_flag(trt.BuilderFlag.FP16))

    # 加载推理引擎，返回ICudaEngine对象
    with open(trt_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # with trt.Builder(logger) as builder:
    #     config = builder.create_builder_config()

    print("2", config.get_flag(trt.BuilderFlag.FP16))
    # model_trt = TRTModule(engine=engine, input_names=["images"], output_names=['991', '1000', '1009'])
    # model_trt = TRTModule(engine=engine, input_names=["images"], output_names=['1964', 'onnx::Slice_1007', 'onnx::Slice_1333', 'onnx::Slice_1658'])  # with grid

    model_trt = TRTModule(engine=engine, input_names=["images"],
                      output_names=['1312', 'onnx::Slice_355', 'onnx::Slice_681', 'onnx::Slice_1006'])  # with grid
    model_trt.to(device)
    # model_trt.half()

    in_tensor = torch.ones(4, 3, 768, 1280).to(device)
    # in_tensor = in_tensor.half().to(device)
    # in_tensor = in_tensor.to(torch.float16)

    outputs = model_trt(in_tensor)
    for output in outputs:
        print(output.shape)


def debug_load_torch2trt(torch2trt_weight_path, device="cuda:0", use_fp16=True):
    # torch2trt_weight_path = "/home/pxn-lyj/Egolee/programs/yolov7-face_liyj/local_files/yolov7-tiny_trt_batch_4.pth"
    model_torch2tr = TRTModule()
    model_torch2tr.load_state_dict(torch.load(torch2trt_weight_path, map_location=device))
    model_torch2tr.eval()

    in_tensor = torch.ones(4, 3, 768, 1280).to(device)
    if use_fp16:
        in_tensor = in_tensor.half().to(device)

    outs = model_torch2tr(in_tensor)
    for out in outs:
        print(out.shape, out[0, 0, 0])


if __name__ == "__main__":
    print("Start")
    # export onnx
    # 采用debug_export.py脚本进行到处
    # --weights /home/pxn-lyj/Egolee/programs/yolov7-face/local_files/yolov7-lite-s.pt
    # --grid                # 会将anchor也导出来,如果就需要进行进一步的解析
    # --batch-size 4        # 到处的batch-size
    # --img-size  768 1280  # 导出的shape
    # --use_fp16            # 是否进行fp16的导出

    # 采用pip的方式进行安装也可以
    # tensorrt version:  pip install tensorrt==8.6.1.post1

    # 将onnx保存为engine,保存同样的文件夹下
    onnx_model_path = "/home/pxn-lyj/Egolee/programs/yolov7-face_liyj/local_files/yolov7-tiny.onnx"
    trt_engine_path = onnx_model_path.replace(".onnx", ".trt")

    # 保存和加载trt-engine
    debug_save_trt_engine(onnx_model_path, trt_engine_path, fp16_mode=True)
    load_trt_engine(trt_engine_path, device="cuda:0")

    # 直接加载torch2trt保存的weight
    # torch2trt_weight_path = "/home/pxn-lyj/Egolee/programs/yolov7-face_liyj/local_files/yolov7-tiny_trt_batch_4.pth"
    # debug_load_torch2trt(torch2trt_weight_path, device="cuda:0", use_fp16=True)
    print("End")