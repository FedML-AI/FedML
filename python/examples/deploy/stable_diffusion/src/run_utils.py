import argparse
import os
from utilities import TRT_LOGGER, add_arguments
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", "/data/scalellm_share_dir")
def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Txt2Img Demo", conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--version', type=str, default="xl-1.0", choices=["xl-1.0"], help="Version of Stable Diffusion XL")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")

    parser.add_argument('--scheduler', type=str, default="DDIM", choices=["PNDM", "LMSD", "DPM", "DDIM", "EulerA"], help="Scheduler for diffusion process")

    parser.add_argument('--onnx-base-dir', default='onnx_xl_base', help="Directory for SDXL-Base ONNX models")
    parser.add_argument('--onnx-refiner-dir', default='onnx_xl_refiner', help="Directory for SDXL-Refiner ONNX models")
    parser.add_argument('--engine-base-dir', default=LOCAL_ROOT + '/Stable-Diffusion/engine_xl_base', help="Directory for SDXL-Base TensorRT engines")
    parser.add_argument('--engine-refiner-dir', default=LOCAL_ROOT + '/Stable-Diffusion/engine_xl_refiner', help="Directory for SDXL-Refiner TensorRT engines")

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    print(args)