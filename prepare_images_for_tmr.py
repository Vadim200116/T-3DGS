import os
import glob
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

def resize_image(image_path, scale_factor):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        resolution = (round(width / scale_factor), round(height / scale_factor))

        img = img.resize(resolution)
        return img


def rename_and_copy_images(src_dir, dest_dir, scale_factor):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    images = glob.glob(os.path.join(src_dir, "*.JPG"))

    images = sorted(filter(lambda x: x.find("clutter") != -1, images))
    for i, image_path in tqdm(enumerate(images)):
        new_file_name = f"{i:05d}.jpg"
        new_file_path = os.path.join(dest_dir, new_file_name)

        resized_img = resize_image(image_path, scale_factor)
        resized_img.save(new_file_path, "JPEG")

def main():
    parser = ArgumentParser(description="SAM2 image processing script parameters")
    
    parser.add_argument(
        "--source_path",
        "-s",
        type=str,
        required=True,
        help="Source directory containing the images"
    )
    
    parser.add_argument(
        "--target_path",
        "-t",
        type=str,
        required=True,
        help="Target directory for processed images"
    )
    
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=1.0,
        help="Scale factor for image resolution (default: 1.0)"
    )

    args = parser.parse_args()
    rename_and_copy_images(args.source_path, args.target_path, args.resolution)

if __name__ == "__main__":
    main()
