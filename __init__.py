from .nodes import ImagePathLoader, UNOParams, UNOGenerator, ImageConcat, ImageSave, ConfigSave

NODE_CLASS_MAPPINGS = {
    "ImagePathLoader": ImagePathLoader,
    "UNOParams": UNOParams,
    "UNOGenerator": UNOGenerator,
    "ImageConcat": ImageConcat,
    "ImageSave": ImageSave,
    "ConfigSave": ConfigSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePathLoader": "Image Path Loader",
    "UNOParams": "UNO Params",
    "UNOGenerator": "UNO Generator",
    "ImageConcat": "Image Concat",
    "ImageSave": "Image Save",
    "ConfigSave": "Config Save",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
