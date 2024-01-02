import run_utils
import base64
import tensorrt as trt

from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
from engine_downloader import download_engine
from utilities import TRT_LOGGER, add_arguments

from txt2img_xl_pipeline import Txt2ImgXLPipeline
from img2img_xl_pipeline import Img2ImgXLPipeline
from cuda import cudart


class StableDiffusionBot(FedMLPredictor):
    def __init__(self):
        super().__init__()
        self.args = run_utils.parseArgs()
        download_engine("stable-diffusion")

        self.args.prompt = ["test_prompt"]
        self.args.negative_prompt = ["test_negative_prompt"]
        if not isinstance(self.args.prompt, list):
            raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(self.args.prompt)}")
        prompt = self.args.prompt * self.args.repeat_prompt

        if not isinstance(self.args.negative_prompt, list):
            raise ValueError(f"`--negative-prompt` must be of type `str` or `str` list, but is {type(self.args.negative_prompt)}")
        if len(self.args.negative_prompt) == 1:
            negative_prompt = self.args.negative_prompt * len(prompt)
        else:
            negative_prompt = self.args.negative_prompt

        # Validate image dimensions
        self.args.image_height = self.args.height
        self.args.image_width = self.args.width
        image_height = self.args.height
        image_width = self.args.width
        if image_height % 8 != 0 or image_width % 8 != 0:
            raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")

        # Register TensorRT plugins
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        max_batch_size = 16
        # FIXME VAE build fails due to element limit. Limitting batch size is WAR
        if self.args.build_dynamic_shape or image_height > 512 or image_width > 512:
            max_batch_size = 4

        batch_size = 1
        if batch_size > max_batch_size:
            raise ValueError(f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4")

        if self.args.use_cuda_graph and (not self.args.build_static_batch or self.args.build_dynamic_shape):
            raise ValueError(f"Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`")

        def init_pipeline(pipeline_class, refinfer, onnx_dir, engine_dir):
            # Initialize demo
            demo = pipeline_class(
                scheduler=self.args.scheduler,
                denoising_steps=self.args.denoising_steps,
                output_dir=self.args.output_dir,
                version=self.args.version,
                hf_token=self.args.hf_token,
                verbose=self.args.verbose,
                nvtx_profile=self.args.nvtx_profile,
                max_batch_size=max_batch_size,
                use_cuda_graph=self.args.use_cuda_graph,
                refiner=refinfer,
                framework_model_dir=self.args.framework_model_dir)

            # Load TensorRT engines and pytorch modules
            demo.loadEngines(engine_dir, self.args.framework_model_dir, onnx_dir, self.args.onnx_opset,
                opt_batch_size=len(prompt), opt_image_height=image_height, opt_image_width=image_width, \
                force_export=self.args.force_onnx_export, force_optimize=self.args.force_onnx_optimize, \
                force_build=self.args.force_engine_build,
                static_batch=self.args.build_static_batch, static_shape=not self.args.build_dynamic_shape, \
                enable_refit=self.args.build_enable_refit, enable_preview=self.args.build_preview_features, \
                enable_all_tactics=self.args.build_all_tactics, \
                timing_cache=self.args.timing_cache, onnx_refit_dir=self.args.onnx_refit_dir)
            return demo   

        self.demo_base = init_pipeline(Txt2ImgXLPipeline, False, self.args.onnx_base_dir, self.args.engine_base_dir)
        self.demo_refiner = init_pipeline(Img2ImgXLPipeline, True, self.args.onnx_refiner_dir, self.args.engine_refiner_dir)
        max_device_memory = max(self.demo_base.calculateMaxDeviceMemory(), self.demo_refiner.calculateMaxDeviceMemory())
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.demo_base.activateEngines(shared_device_memory)
        self.demo_refiner.activateEngines(shared_device_memory)
        self.demo_base.loadResources(image_height, image_width, batch_size, self.args.seed)
        self.demo_refiner.loadResources(image_height, image_width, batch_size, self.args.seed)

    def run_sd_xl_inference(self, warmup=False, verbose=False):
        images, _, time_base = self.demo_base.infer(self.args.prompt, self.args.negative_prompt, 
                                                 self.args.image_height, self.args.image_width, 
                                                 warmup=warmup, verbose=verbose, seed=self.args.seed, return_type="latents")
        images, paths, time_refiner = self.demo_refiner.infer(self.args.prompt, self.args.negative_prompt, 
                                                       images, self.args.image_height, self.args.image_width, 
                                                       warmup=warmup, verbose=verbose, seed=self.args.seed)
        return images, paths, time_base + time_refiner
        
    def predict(self, request: dict, header=None):
        # --width 1024   --height 1024   --denoising-steps 30
        args = self.args
        input_dict = request
        prompt: str = input_dict.get("text", "").strip()

        self.args.prompt = [prompt] # Default batch size is 1
        
        images, paths, pipeline_time = self.run_sd_xl_inference(warmup=False, verbose=args.verbose)
        print(f"paths: {paths}")
        
        if len(prompt) == 0:
            response_text = "<received empty input; no response generated.>"
        else:
            if header == "image/png":
                return str(paths[0])
            else:
                with open(paths[0], "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    return encoded_string


if __name__ == "__main__":
    chatbot = StableDiffusionBot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
    print("Release resources ...")
    chatbot.demo_base.teardown()
    chatbot.demo_refiner.teardown()
    print("Done.")
