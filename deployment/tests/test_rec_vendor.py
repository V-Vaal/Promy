from api.models.ocr_model import MODEL_DIR, _build_local_infer_args, _resolve_rec_image_shape
from api.vendor.rec_infer_runner import TextRecognizer


def test_build_local_infer_args_matches_nb3_contract():
    args = _build_local_infer_args(use_gpu=False)

    assert args.rec_algorithm == "CRNN"
    assert args.rec_model_dir == str(MODEL_DIR)
    assert args.rec_char_dict_path == str(MODEL_DIR / "en_dict.txt")
    assert args.rec_image_shape == "3,32,100"
    assert args.rec_batch_num == 12
    assert args.show_log is False
    assert args.warmup is False
    assert args.enable_mkldnn is False


def test_resolve_rec_image_shape_reads_inference_yml():
    assert _resolve_rec_image_shape(MODEL_DIR) == "3,32,100"


def test_text_recognizer_initializes_from_vendorized_stack():
    recognizer = TextRecognizer(_build_local_infer_args(use_gpu=False))

    assert recognizer.rec_algorithm == "CRNN"
    assert recognizer.rec_image_shape == [3, 32, 100]
    assert recognizer.rec_batch_num == 12
    assert recognizer.postprocess_name == "CTCLabelDecode"
