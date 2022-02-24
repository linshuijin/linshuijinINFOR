import argparse
import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Onnx Model FP32 to FP16',
                                     add_help=add_help)
    parser.add_argument('--model',
                        # default='inception_v3',
                        default='resnet18',
                        help='model')
    return parser


def main(args):
    model_name_fp32 = "{}_fp32.onnx".format(args.model)
    model_name_fp16 = "{}_fp16.onnx".format(args.model)
    onnx_model = load_model(model_name_fp32)
    onnx.checker.check_model(onnx_model)
    print("FP32 ==> Passed")
    new_onnx_model = convert_float_to_float16(onnx_model)
    onnx.checker.check_model(new_onnx_model)
    print("FP16 ==> Passed")
    save_model(new_onnx_model, model_name_fp16)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
