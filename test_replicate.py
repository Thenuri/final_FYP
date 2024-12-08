import replicate
import os

# If using python-dotenv
from dotenv import load_dotenv
load_dotenv()

if 'REPLICATE_API_TOKEN' not in os.environ:
    print("REPLICATE_API_TOKEN not found. Please set it as an environment variable.")
    exit()

try:
    # Define the input parameters for the model
    inputs = {
        "model": "dev",
        "prompt": "THENUSAN, full body image of women wearing Minimal Embroidery embroidered tunic",
        "lora_scale": 1,
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "guidance_scale": 3.5,
        "output_quality": 90,
        "prompt_strength": 0.8,
        "extra_lora_scale": 1,
        "num_inference_steps": 28
    }

    # Run the model using replicate.run
    output = replicate.run(
        "thenuri/flux-full-body-image-generation:30176a44e38cddb6dd3adb8f191f6da2f6ff5cce5dc837201855c8d2a6c95e24",
        input=inputs
    )

    # The output is a list of image URLs
    if output and isinstance(output, list) and len(output) > 0:
        image_url = output[0]
        print(f"Image URL: {image_url}")
    else:
        print("No valid image URL returned from the API.")
        print(f"Output received: {output}")

except Exception as e:
    print(f"An error occurred during image generation: {e}")
    import traceback
    print(traceback.format_exc())
