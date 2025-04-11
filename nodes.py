import os
import dataclasses
from typing import Literal
from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
import json
import itertools

from .src.flux.pipeline import UNOPipeline, preprocess_ref


def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im


@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 512
    height: int = 512
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 4
    seed: int = 3407
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 512
    data_resolution: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'


def main(args: InferenceArgs):
    accelerator = Accelerator()

    pipeline = UNOPipeline(
        args.model_type,
        accelerator.device,
        args.offload,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank
    )

    assert args.prompt is not None or args.eval_json_path is not None, \
        "Please provide either prompt or eval_json_path"
    
    if args.eval_json_path is not None:
        with open(args.eval_json_path, "rt") as f:
            data_dicts = json.load(f)
        data_root = os.path.dirname(args.eval_json_path)
    else:
        data_root = "./"
        data_dicts = [{"prompt": args.prompt, "image_paths": args.image_paths}]

    for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
        if (i * args.num_images_per_prompt + j) % accelerator.num_processes != accelerator.process_index:
            continue

        ref_imgs = [
            Image.open(os.path.join(data_root, img_path))
            for img_path in data_dict["image_paths"]
        ]
        if args.ref_size==-1:
            args.ref_size = 512 if len(ref_imgs)==1 else 320

        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]

        image_gen = pipeline(
            prompt=data_dict["prompt"],
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed + j,
            ref_imgs=ref_imgs,
            pe=args.pe,
        )
        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        os.makedirs(args.save_path, exist_ok=True)
        image_gen.save(os.path.join(args.save_path, f"{i}_{j}.png"))

        # save config and image
        args_dict = vars(args)
        args_dict['prompt'] = data_dict["prompt"]
        args_dict['image_paths'] = data_dict["image_paths"]
        with open(os.path.join(args.save_path, f"{i}_{j}.json"), 'w') as f:
            json.dump(args_dict, f, indent=4)        


class ImagePathLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_paths": ("STRING", {
                    "default": "assets/image1.png,assets/image2.png",
                    "multiline": False
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image_list", "path_list",)
    FUNCTION = "load_images"
    CATEGORY = "UNO/Preprocess"

    def load_images(self, image_paths):
        paths = [p.strip() for p in image_paths.split(",")]
        images = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            images.append(img)
        return (images, paths)


class UNOParams:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A clock on the beach is under a red sun umbrella"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 3407, "min": 0, "max": 999999}),
                "ref_size": ("INT", {"default": -1, "min": -1, "max": 1024}),
                "pe": (["d", "h", "w", "o"],),
                "concat_refs": (["enable", "disable"],),
                "save_path": ("STRING", {"default": "output/inference"}),
            }
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "pack_params"
    CATEGORY = "UNO/Params"

    def pack_params(self, **kwargs):
        return (kwargs,)


class UNOGenerator:
    def __init__(self):
        self.accelerator = Accelerator()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "params": ("DICT",),
                "ref_images": ("IMAGE",),
                "model_type": (["flux-dev", "flux-dev-fp8", "flux-schnell"],),
                "offload": (["enable", "disable"],),
                "only_lora": (["enable", "disable"],),
                "lora_rank": ("INT", {"default": 512, "min": 1, "max": 2048})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "UNO/Inference"

    def run_inference(self, params, ref_images, model_type, offload, only_lora, lora_rank):
        device = self.accelerator.device
        pipeline = UNOPipeline(
            model_type,
            device,
            offload == "enable",
            only_lora=only_lora == "enable",
            lora_rank=lora_rank
        )
        if params.get("ref_size", -1) == -1:
            ref_size = 512 if len(ref_images) == 1 else 320
        else:
            ref_size = params["ref_size"]

        ref_imgs = [preprocess_ref(img, ref_size) for img in ref_images]

        image_gen = pipeline(
            prompt=params["prompt"],
            width=params["width"],
            height=params["height"],
            guidance=params["guidance"],
            num_steps=params["num_steps"],
            seed=params["seed"],
            ref_imgs=ref_imgs,
            pe=params["pe"]
        )
        
        return (image_gen,)


class ImageConcat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "main_image": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "concat_refs": (["enable", "disable"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat_images"
    CATEGORY = "UNO/Postprocess"

    def concat_images(self, main_image, ref_images, concat_refs):
        if concat_refs == "disable":
            return (main_image,)
        images = [main_image] + ref_images
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            new_im.paste(img, (x_offset, 0))
            x_offset += img.size[0]
            
        return (new_im,)


class ImageSave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "save_path": ("STRING", {"default": "output/inference"}),
                "filename_prefix": ("STRING", {"default": "generated"})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_image"
    CATEGORY = "UNO/Postprocess"

    def save_image(self, image, save_path, filename_prefix):
        os.makedirs(save_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(save_path, f"{filename_prefix}_{timestamp}.png")
        image.save(file_path)
        
        return (file_path,)


class ConfigSave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "params": ("DICT",),
                "path_list": ("STRING",),
                "save_path": ("STRING", {"default": "output/inference"}),
                "filename_prefix": ("STRING", {"default": "config"})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_config"
    CATEGORY = "UNO/Postprocess"

    def save_config(self, params, path_list, save_path, filename_prefix):
        os.makedirs(save_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(save_path, f"{filename_prefix}_{timestamp}.json")
        params["image_paths"] = path_list
        with open(file_path, "w") as f:
            json.dump(params, f, indent=4)
            
        return (file_path,)


