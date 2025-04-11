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


class LoadImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refer_image_paths": ("STRING", {"multiline": True, "default": "["./assets/examples", ]"}),
                "height": ("INT", {"default": 480}),
                "width": ("INT", {"default": 832}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("image_paths")
    FUNCTION = "process_images"
    CATEGORY = "UNO"

    def process_images(self, refer_image_paths, height, width, device):
        

        return (image_paths,)



if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
  
