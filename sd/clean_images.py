import os
from PIL import Image
from pathlib import Path

def clean_image_dataset(dataset_path):
    """
    Iterate through image dataset, delete corrupted images and invalid filenames, return count of valid images.
    
    Args:
        dataset_path: Path to the image directory
        
    Returns:
        tuple: (valid_count, deleted_count, invalid_name_count, total_count)
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Directory {dataset_path} does not exist")
        return 0, 0, 0, 0
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    valid_count = 0
    deleted_count = 0
    invalid_name_count = 0
    total_count = 0
    
    # Get all image files
    image_files = [f for f in dataset_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    total_count = len(image_files)
    print(f"Found {total_count} image files to check...")
    
    for img_path in image_files:
        # Skip hidden files and system files
        if img_path.name.startswith('.'):
            print(f"Deleting system file: {img_path.name}")
            img_path.unlink()
            continue
        
        # Check if filename (without extension) is a valid integer
        filename_without_ext = img_path.stem  # e.g., "1458(1)" from "1458(1).jpg"
        
        try:
            int(filename_without_ext)
            is_valid_name = True
        except ValueError:
            is_valid_name = False
        
        if not is_valid_name:
            print(f"Deleting invalid filename: {img_path.name}")
            try:
                img_path.unlink()
                invalid_name_count += 1
                deleted_count += 1
            except Exception as del_error:
                print(f"  Failed to delete {img_path.name}: {del_error}")
            continue
        
        # Now check if image is valid
        try:
            # Try to open and verify the image
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
            
            # Reopen to check if it can be loaded (verify closes the file)
            with Image.open(img_path) as img:
                img.load()  # Actually load the image data
            
            valid_count += 1
            
        except Exception as e:
            print(f"Deleting corrupted image: {img_path.name} - Error: {e}")
            try:
                img_path.unlink()  # Delete the file
                deleted_count += 1
            except Exception as del_error:
                print(f"  Failed to delete {img_path.name}: {del_error}")
    
    print(f"\n=== Results ===")
    print(f"Total images checked: {total_count}")
    print(f"Valid images: {valid_count}")
    print(f"Invalid filenames deleted: {invalid_name_count}")
    print(f"Corrupted images deleted: {deleted_count - invalid_name_count}")
    print(f"Total deleted: {deleted_count}")
    
    return valid_count, deleted_count, invalid_name_count, total_count


if __name__ == "__main__":
    # Use relative path
    dataset_path = "../image_from_url"
    
    # Or use absolute path
    # dataset_path = "/home/ubuntu/BayesDiff_18794/image_from_url"
    
    valid, deleted, invalid_names, total = clean_image_dataset(dataset_path)
    
    print(f"\nDataset is clean! {valid} valid images remaining.")
