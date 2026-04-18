import argparse
import math
import os
import time

import cv2
import numpy as np

from api.vendor.rec_infer_utility import create_predictor, load_config
from api.vendor.rec_logging import get_logger
from api.vendor.rec_postprocess import CTCLabelDecode

os.environ["FLAGS_allocator_strategy"] = "auto_growth"


class TextRecognizer:
    def __init__(self, args: argparse.Namespace, logger=None) -> None:
        if os.path.exists(f"{args.rec_model_dir}/inference.yml"):
            model_config = load_config(f"{args.rec_model_dir}/inference.yml")
            model_name = model_config.get("Global", {}).get("model_name", "")
            if model_name and model_name not in [
                "PP-OCRv5_mobile_rec",
                "PP-OCRv5_server_rec",
                "korean_PP-OCRv5_mobile_rec",
                "eslav_PP-OCRv5_mobile_rec",
                "latin_PP-OCRv5_mobile_rec",
                "en_PP-OCRv5_mobile_rec",
                "th_PP-OCRv5_mobile_rec",
                "el_PP-OCRv5_mobile_rec",
            ]:
                raise ValueError(f"{model_name} is not supported.")

        self.logger = logger or get_logger()
        self.rec_image_shape = [int(value) for value in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.postprocess_op = CTCLabelDecode(
            character_dict_path=args.rec_char_dict_path,
            use_space_char=args.use_space_char,
        )
        self.postprocess_name = "CTCLabelDecode"
        self.predictor, self.input_tensor, self.output_tensors, self.config = create_predictor(
            args,
            "rec",
            self.logger,
        )
        self.benchmark = args.benchmark
        self.use_onnx = args.use_onnx
        self.return_word_box = args.return_word_box

    def resize_norm_img(self, image: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_channels, image_height, image_width = self.rec_image_shape
        assert img_channels == image.shape[2]
        image_width = int(image_height * max_wh_ratio)
        if self.use_onnx:
            width = self.input_tensor.shape[3:][0]
            if width is not None and not isinstance(width, str) and width > 0:
                image_width = width

        height, width = image.shape[:2]
        ratio = width / float(height)
        resized_width = image_width if math.ceil(image_height * ratio) > image_width else int(
            math.ceil(image_height * ratio)
        )
        resized_image = cv2.resize(image, (resized_width, image_height))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_image = np.zeros((img_channels, image_height, image_width), dtype=np.float32)
        padding_image[:, :, 0:resized_width] = resized_image
        return padding_image

    def __call__(self, image_list: list[np.ndarray]) -> tuple[list[tuple[str, float]], float]:
        start_time = time.time()
        width_indices = [[image.shape[1] / float(image.shape[0]), index] for index, image in enumerate(image_list)]
        width_indices = np.array(sorted(width_indices, key=lambda item: item[0]))
        recognition_results: list[tuple[str, float]] = [["", 0.0]] * len(image_list)

        for begin_index in range(0, len(image_list), self.rec_batch_num):
            end_index = min(len(image_list), begin_index + self.rec_batch_num)
            max_wh_ratio = self.rec_image_shape[2] / self.rec_image_shape[1]
            wh_ratio_list: list[float] = []
            norm_image_batch = []
            for sorted_index in range(begin_index, end_index):
                image_index = int(width_indices[sorted_index][1])
                wh_ratio = image_list[image_index].shape[1] / float(image_list[image_index].shape[0])
                wh_ratio_list.append(wh_ratio)
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                norm_image = self.resize_norm_img(image_list[image_index], max_wh_ratio)
                norm_image_batch.append(norm_image[np.newaxis, :])

            batch = np.concatenate(norm_image_batch).copy()
            self.input_tensor.copy_from_cpu(batch)
            self.predictor.run()
            outputs = [output_tensor.copy_to_cpu() for output_tensor in self.output_tensors]
            predictions = outputs if len(outputs) != 1 else outputs[0]
            if self.postprocess_name != "CTCLabelDecode":
                raise ValueError(f"Unsupported postprocess: {self.postprocess_name}")
            batch_results = self.postprocess_op(predictions)
            for result_index, result in enumerate(batch_results):
                image_index = int(width_indices[begin_index + result_index][1])
                recognition_results[image_index] = result

        return recognition_results, time.time() - start_time
