import os
import json
import numpy as np
import tifffile as tiff
import argparse
import random
from pathlib import Path
from PIL import Image
from shapely.geometry import shape
from skimage.draw import polygon
from tqdm import tqdm

def extract_nuclei_counts(geojson_path):
    """Extract nuclei counts from nuclei geojson file."""
    nuclei_counts = {}
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    for feature in data['features']:
        if feature['geometry']['type'] == 'Polygon':
            nuclei_type = feature['properties'].get('classification', {}).get('name', 'unknown')
            nuclei_counts[nuclei_type] = nuclei_counts.get(nuclei_type, 0) + 1
    
    return nuclei_counts

def extract_tissue_type(geojson_path):
    """Extract tissue type from tissue geojson file."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    tissue_types = {feature['properties'].get('classification', {}).get('name', 'unknown') for feature in data['features'] if feature['geometry']['type'] == 'Polygon'}
    return tissue_types.pop() if tissue_types else 'unknown'

def create_segmentation_mask(geojson_path, image_size=(1024, 1024)):
    """Generate segmentation mask from geojson annotations."""
    mask = np.zeros(image_size, dtype=np.uint16)
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    instance_idx = 1
    for feature in data['features']:
        if feature['geometry']['type'] == 'Polygon':
            polygon_coords = np.array(feature['geometry']['coordinates'][0])
            row_coords, col_coords = polygon(polygon_coords[:, 1], polygon_coords[:, 0], shape=image_size)
            mask[row_coords, col_coords] = instance_idx
            instance_idx += 1
    
    return mask

def process_fold(fold, input_tiff_path, input_nuclei_path, input_tissue_path, output_path):
    """Process images and labels for a given fold."""
    output_fold_path = Path(output_path) / f"fold{fold}"
    output_fold_path.mkdir(parents=True, exist_ok=True)
    (output_fold_path / "images").mkdir(exist_ok=True)
    (output_fold_path / "labels").mkdir(exist_ok=True)
    
    cell_counts = []
    tissue_data = [["img", "type"]]  # Header for types.csv
    nuclei_types_set = set()
    
    tiff_files = sorted(Path(input_tiff_path).glob("*.tif"))
    tiff_files = [file for idx, file in enumerate(tiff_files) if idx % 3 == fold]
    
    for index, file in tqdm(enumerate(tiff_files), total=len(tiff_files), desc=f"Processing fold{fold}"):
        sample_name = f"{fold}_{index}"  # New file naming format
        
        # Load and save image
        img = tiff.imread(file)
        img = Image.fromarray(img)
        img.save(output_fold_path / "images" / f"{sample_name}.png")
        
        # Process corresponding geojson files
        nuclei_geojson_path = Path(input_nuclei_path) / f"{file.stem}_nuclei.geojson"
        tissue_geojson_path = Path(input_tissue_path) / f"{file.stem}_tissue.geojson"
        
        nuclei_counts = extract_nuclei_counts(nuclei_geojson_path) if nuclei_geojson_path.exists() else {}
        tissue_type = extract_tissue_type(tissue_geojson_path) if tissue_geojson_path.exists() else 'unknown'
        
        tissue_data.append([f"{sample_name}.png", tissue_type])
        nuclei_types_set.update(nuclei_counts.keys())
        
        # Generate segmentation mask from nuclei annotations
        mask = create_segmentation_mask(nuclei_geojson_path) if nuclei_geojson_path.exists() else np.zeros((1024, 1024))
        np.save(output_fold_path / "labels" / f"{sample_name}.npy", {"inst_map": mask, "type_map": np.zeros_like(mask, dtype=np.int32)})
    
    # Prepare cell_count.csv with headers
    nuclei_types_list = sorted(nuclei_types_set)
    cell_counts.append(["Image"] + nuclei_types_list)
    
    for index, file in tqdm(enumerate(tiff_files), total=len(tiff_files), desc=f"Saving CSV for fold{fold}"):
        sample_name = f"{fold}_{index}"
        nuclei_counts = extract_nuclei_counts(Path(input_nuclei_path) / f"{file.stem}_nuclei.geojson") if (Path(input_nuclei_path) / f"{file.stem}_nuclei.geojson").exists() else {}
        cell_counts.append([f"{sample_name}.png"] + [nuclei_counts.get(nuc_type, 0) for nuc_type in nuclei_types_list])
    
    # Save CSV files
    np.savetxt(output_fold_path / "cell_count.csv", cell_counts, delimiter=",", fmt="%s")
    np.savetxt(output_fold_path / "types.csv", tissue_data, delimiter=",", fmt="%s")

def main():
    parser = argparse.ArgumentParser(description="Prepare PUMA dataset")
    parser.add_argument("--input_tiff_path", type=str, required=True, help="Path to the TIFF images folder")
    parser.add_argument("--input_nuclei_path", type=str, required=True, help="Path to the nuclei annotations folder")
    parser.add_argument("--input_tissue_path", type=str, required=True, help="Path to the tissue annotations folder")
    parser.add_argument("--output_path", type=str, required=True, help="Path to store processed dataset")
    args = parser.parse_args()
    
    for fold in [0, 1, 2]:
        process_fold(fold, args.input_tiff_path, args.input_nuclei_path, args.input_tissue_path, args.output_path)

if __name__ == "__main__":
    main()
