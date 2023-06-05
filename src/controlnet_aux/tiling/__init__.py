from PIL import Image


class TilingDetector:
    def __call__(self, input_image: Image.Image, resolution: int = 1024):
        input_image = input_image.convert("RGB")
        original_width, original_height = input_image.size

        # Calculate the scale factor based on the desired resolution
        scale_factor = float(resolution) / min(original_height, original_width)

        # Scale the width and height of the image
        scaled_height = original_height * scale_factor
        scaled_width = original_width * scale_factor

        # Round the scaled dimensions to the nearest multiple of 64
        scaled_height = int(round(scaled_height / 64.0)) * 64
        scaled_width = int(round(scaled_width / 64.0)) * 64

        # Resize the image using Lanczos interpolation
        resized_image = input_image.resize(
            (scaled_width, scaled_height), resample=Image.LANCZOS
        )

        return resized_image
