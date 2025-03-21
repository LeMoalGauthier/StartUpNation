import os
import csv
import torch
from pytorch_fid import fid_score
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tempfile

def calculate_fid(generated_image_path, reference_image_path):
    """Calculates FID score between a single generated image and its reference image."""
    try:
        # Open and resize images if necessary
        generated_image = Image.open(generated_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")
        
        if generated_image.size != reference_image.size:
            print(f"Resizing {generated_image_path} to match {reference_image_path}")
            generated_image = generated_image.resize(reference_image.size)

        # Create temporary directories for FID calculation
        with tempfile.TemporaryDirectory() as gen_dir, tempfile.TemporaryDirectory() as ref_dir:
            # Save images to temporary directories
            gen_path = os.path.join(gen_dir, "image.png")
            ref_path = os.path.join(ref_dir, "image.png")
            generated_image.save(gen_path)
            reference_image.save(ref_path)

            # Calculate FID with proper directory structure
            fid = fid_score.calculate_fid_given_paths(
                paths=[gen_dir, ref_dir],
                batch_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dims=2048,
                num_workers=0  # Sometimes better to set to 0 for stability
            )
            
        return fid
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None


def calculate_clip_score(prompt, image_path):
    """Calculates CLIP similarity score between a prompt and an image."""
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        return logits_per_image.item()
    except Exception as e:
        print(f"Error calculating CLIP score: {e}")
        return None

def evaluate_images(output_dir):
    """Evaluates images in the output directory using FID and CLIP scores."""
    evaluation_results = []
    
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Read the prompt file
        prompt_file = os.path.join(folder_path, "prompt")
        if not os.path.exists(prompt_file):
            print(f"Prompt file not found in {folder_path}")
            continue
        
        with open(prompt_file, "r") as f:
            prompt = f.read().strip()
        
        # Find reference and generated images
        reference_image_path = None
        for image_file in os.listdir(folder_path):
            if image_file in ["reference.jpeg", "reference.jpg"]:
                reference_image_path = os.path.join(folder_path, image_file)
                break
        
        if not reference_image_path:
            print(f"No reference image found in {folder_path}")
            continue
        
        for image_file in os.listdir(folder_path):
            if image_file.endswith((".png", ".jpg", ".jpeg")) and image_file not in ["reference.jpeg", "reference.jpg"]:
                generated_image_path = os.path.join(folder_path, image_file)
                generated_image_path = os.path.join(folder_path, image_file)
                
                # Calculate FID score
                fid_score_value = calculate_fid(generated_image_path, reference_image_path)
                
                # Calculate CLIP score
                clip_score_value = calculate_clip_score(prompt, generated_image_path)
                
                evaluation_results.append({
                    "Folder": folder,
                    "Image": image_file,
                    "Prompt": prompt,
                    "FID": fid_score_value,
                    "CLIP": clip_score_value
                })
    
    # Save results to CSV
    csv_file = "csv/evaluation.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Folder", "Image", "Prompt", "FID", "CLIP"])
        writer.writeheader()
        writer.writerows(evaluation_results)
    
    print(f"Evaluation results saved to {csv_file}")

if __name__ == "__main__":
    output_directory = "outputs"
    
    evaluate_images(output_directory)
