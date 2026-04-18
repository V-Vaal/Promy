import argparse
import os

import numpy as np
import paddle
import yaml
from paddle import inference

from api.vendor.rec_logging import get_logger


def str2bool(value: str) -> bool:
    return value.lower() in ("true", "yes", "t", "y", "1")


def init_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--use_mlu", type=str2bool, default=False)
    parser.add_argument("--use_metax_gpu", type=str2bool, default=False)
    parser.add_argument("--use_gcu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--rec_algorithm", type=str, default="SVTR_LCNet")
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--rec_char_dict_path", type=str, default="./ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=None)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")
    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)
    parser.add_argument("--return_word_box", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)
    return parser


def load_config(file_path: str) -> dict:
    _, extension = os.path.splitext(file_path)
    if extension not in [".yml", ".yaml"]:
        raise ValueError(f"only support yaml files for now, got {file_path}")
    with open(file_path, "rb") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def get_output_tensors(args: argparse.Namespace, mode: str, predictor) -> list:
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and args.rec_algorithm in ["CRNN", "SVTR_LCNet", "SVTR_HGNet"]:
        output_name = "softmax_0.tmp_0"
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
    for output_name in output_names:
        output_tensors.append(predictor.get_output_handle(output_name))
    return output_tensors


def get_infer_gpuid() -> int:
    logger = get_logger()
    gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = gpu_id_str.split(",")
    logger.warning("The first GPU is used for inference by default, GPU ID: %s", gpu_ids[0])
    return int(gpu_ids[0])


def create_predictor(args: argparse.Namespace, mode: str, logger):
    model_dir = args.rec_model_dir if mode == "rec" else None
    if model_dir is None:
        logger.info("not find %s model file path %s", mode, model_dir)
        raise ValueError(f"Missing model directory for mode {mode}")

    file_names = ["model", "inference"]
    params_file_path = ""
    model_file_path = ""
    for file_name in file_names:
        candidate_params_path = f"{model_dir}/{file_name}.pdiparams"
        if os.path.exists(candidate_params_path):
            params_file_path = candidate_params_path
            json_path = f"{model_dir}/{file_name}.json"
            pdmodel_path = f"{model_dir}/{file_name}.pdmodel"
            if os.path.exists(json_path):
                model_file_path = json_path
            elif os.path.exists(pdmodel_path):
                model_file_path = pdmodel_path
            break

    if not params_file_path:
        raise ValueError(f"not find inference.pdiparams in {model_dir}")
    if not model_file_path:
        raise ValueError(f"neither inference.json nor inference.pdmodel was found in {model_dir}.")

    config = inference.Config(model_file_path, params_file_path)
    if args.use_gpu:
        _ = get_infer_gpuid()
        config.enable_use_gpu(args.gpu_mem, args.gpu_id)
    else:
        config.disable_gpu()
        if args.enable_mkldnn:
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(args.cpu_threads)
        if hasattr(config, "enable_new_ir"):
            config.enable_new_ir()
        if hasattr(config, "enable_new_executor"):
            config.enable_new_executor()

    config.enable_memory_optim()
    config.disable_glog_info()
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.delete_pass("matmul_transpose_reshape_fuse_pass")
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    predictor = inference.create_predictor(config)
    input_tensor = predictor.get_input_handle(predictor.get_input_names()[0])
    output_tensors = get_output_tensors(args, mode, predictor)
    return predictor, input_tensor, output_tensors, config
