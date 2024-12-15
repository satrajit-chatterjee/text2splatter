import base64
import os
import random
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)

prompt = """Your task is to generate 10 prompts for each image that you see. The prompt should be concise and be in the imperative form. 
            For example if you see an image of a black cat playing with a toy, the prompt "Create a black cat playing with a toy" would be a good prompt.
            For each image, generate 10 prompts and then return them in json format like this:
            {
                "image1.jpg": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5", "prompt6", "prompt7", "prompt8", "prompt9", "prompt10"],
            }
"""

def generate_image_captions(image_path: str, base64_encoded_image: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_encoded_image}",
                            "detail": "low"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
        response_format={ "type" : "json_object" }
    )
    return response.choices[0]
  
  

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode("utf-8")
  

def select_images(image_folder: str, num_images: int = 5):
  """For each image in an images folder fetches 5 random ones with png or jpg extension.

  Args:
      images_folder (str): _description_
  Returns:
      dict[str, ]: List of image paths
  """
  image_paths = []
  for image in os.listdir(image_folder):
      if image.endswith(".png") or image.endswith(".jpg"):
          image_paths.append(os.path.join(image_folder, image))
  
  # select n random images
  select_images = random.sample(image_paths, num_images)
  
  return select_images

  
    
    
def main():
    parser = argparse.ArgumentParser(description="Generate image captions using OpenAI API.")
    parser.add_argument("--images_folder", type=str, required=True, help="Path to the folder containing images.")
    args = parser.parse_args()

    images_folder = args.images_folder
    dir_list = os.listdir(images_folder)
    for image_folder in dir_list[:1]:
        image_path = os.path.join(images_folder, image_folder)
        image_paths = select_images(image_path)
        for image_path in image_paths:
            encoded_image = encode_image(image_path)
            image_captions = generate_image_captions(image_folder, encoded_image)
            print(image_captions)

if __name__ == "__main__":
    main()
