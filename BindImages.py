import SimpleITK as sitk
import argparse
import numpy as np
from pathlib import Path

"""
This script is a tool to bind multiple Nifti or DICOM images into a single image by resampling each image to a common image space,
as might be necessary for multi-region or whole-body imaging. Overlapping regions are averaged together.
This version uses SimpleITK for all image processing operations.
Alan McMillan 2023 (abmcmillan@wisc.edu) - Rewritten by Jules 2025

Inputs:
    --input: The input images/directories to combine.
    --output: The output image.
    --as_float: Convert the images to float before combining.
    --interp_type: Interpolation type to use. Default is linear.
    --voxel_size: The voxel size of the output image (three floating point inputs). Default is to infer from the first input.

Outputs:
    The output image.

Example usage:
    python BindImages.py --input image1.nii.gz image2.nii.gz --output bound_image.nii.gz
    python BindImages.py --input dicom_dir1 dicom_dir2 --output bound_image.nii.gz

Requirements:
    - SimpleITK
    - numpy
    - Python 3.6 or higher
"""

# Mapping from string arguments to SimpleITK interpolator constants
interpolator_map = {
    'linear': sitk.sitkLinear,
    'nearestNeighbor': sitk.sitkNearestNeighbor,
    'gaussian': sitk.sitkGaussian,
    'bSpline': sitk.sitkBSpline,
    'cosineWindowedSinc': sitk.sitkCosineWindowedSinc,
    'hammingWindowedSinc': sitk.sitkHammingWindowedSinc,
    'lanczosWindowedSinc': sitk.sitkLanczosWindowedSinc,
    'genericLabel': sitk.sitkLabelGaussian,
}

def read_dicom_series(directory):
    """
    Reads a DICOM series from a directory and returns a SimpleITK image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    if not dicom_names:
        raise ValueError(f"No DICOM series found in directory: {directory}")
    reader.SetFileNames(dicom_names)
    image_sitk = reader.Execute()
    return image_sitk

# Step 0: Parse the command line arguments
parser = argparse.ArgumentParser(description='Combine multiple Nifti or DICOM images into a single image using SimpleITK.')
parser.add_argument('--input', type=str, nargs='+', help='The input images or DICOM directories to combine.', required=True)
parser.add_argument('--output', type=str, help='The output image file.', required=True)
parser.add_argument('--as_float', action='store_true', help='Convert the images to float32 before combining.')
parser.add_argument('--interp_type',
                    choices=list(interpolator_map.keys()),
                    default='linear', help="Interpolation type to use for resampling.")
parser.add_argument('--voxel_size', type=float, nargs=3, help='The voxel size of the output image (three floating point inputs).', required=False)
args = parser.parse_args()

# Check that the input files/directories exist
for fp in args.input:
    path = Path(fp)
    if not (path.is_file() or path.is_dir()):
        raise Exception(f'Input {fp} is not a valid file or directory.')

# Check that the output file does not exist
if Path(args.output).is_file():
    raise Exception(f'Output file {args.output} already exists.')

# Step 1: Load the images
print('loading images...')
images = []
for fp in args.input:
    path = Path(fp)
    if path.is_dir():
        print(f"Reading DICOM series from directory: {fp}")
        images.append(read_dicom_series(fp))
    else:
        print(f"Reading image file: {fp}")
        images.append(sitk.ReadImage(str(fp)))

# Step 2: Determine the spatial extent in world space coordinates
print('computing spatial extents...')
real_world_coords = []
for i, img in enumerate(images):
    print(f'file: {args.input[i]}')
    size = img.GetSize()
    print(f'    shape: {size}')
    print(f'    spacing: {img.GetSpacing()}')
    print(f'    origin: {img.GetOrigin()}')

    # Get the 8 corners of the image in pixel coordinates
    corners_index = []
    for x in [0, size[0]-1]:
        for y in [0, size[1]-1]:
            for z in [0, size[2]-1]:
                corners_index.append((x,y,z))

    # Transform corner indices to physical points
    physical_corners = [img.TransformIndexToPhysicalPoint(c) for c in corners_index]
    real_world_coords.extend(physical_corners)

min_coords = np.min(real_world_coords, axis=0)
max_coords = np.max(real_world_coords, axis=0)
extent_range = max_coords - min_coords

print(f'    min_extent: {min_coords}')
print(f'    max_extent: {max_coords}')
print(f'    extent_range: {extent_range}')

# Step 3: Define the geometry of the output image
if args.voxel_size is None:
    newimg_spacing = images[0].GetSpacing()
    newimg_spacing = tuple(value / 2 for value in newimg_spacing)
    print(f'    new image spacing (inferred from first input, halved): {newimg_spacing}')
else:
    newimg_spacing = tuple(args.voxel_size)

newimg_size = [int(np.ceil(extent_range[i] / newimg_spacing[i])) for i in range(3)]
newimg_origin = tuple(min_coords)
newimg_direction = np.eye(3).flatten().tolist()
output_pixel_type = sitk.sitkFloat32 if args.as_float else images[0].GetPixelID()

# Step 4: Create the output image and resample inputs
print('creating new image and binding...')
resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(newimg_spacing)
resampler.SetOutputOrigin(newimg_origin)
resampler.SetOutputDirection(newimg_direction)
resampler.SetSize(newimg_size)
resampler.SetInterpolator(interpolator_map[args.interp_type])
resampler.SetDefaultPixelValue(0)

# Accumulator for the resampled images and the overlap count
final_image_accumulator = sitk.Image(newimg_size, sitk.sitkFloat32)
final_image_accumulator.SetSpacing(newimg_spacing)
final_image_accumulator.SetOrigin(newimg_origin)
final_image_accumulator.SetDirection(newimg_direction)

overlap_accumulator = sitk.Image(newimg_size, sitk.sitkFloat32)
overlap_accumulator.CopyInformation(final_image_accumulator)

for img in images:
    # Resample image
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampled_img = resampler.Execute(img)
    final_image_accumulator += resampled_img

    # Create a binary mask for overlap counting
    binary_mask = sitk.Image(img.GetSize(), sitk.sitkFloat32)
    binary_mask.CopyInformation(img)
    binary_mask += 1.0 # Fill with 1s

    # Resample mask
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Use nearest neighbor for mask
    resampled_mask = resampler.Execute(binary_mask)
    overlap_accumulator += resampled_mask

# Step 5: Average the overlapping regions
# Avoid division by zero where there is no overlap
overlap_accumulator = sitk.Maximum(overlap_accumulator, 1.0)
final_image = final_image_accumulator / overlap_accumulator

# Cast to original pixel type if not --as_float
if not args.as_float:
    # Get the pixel ID from the first input image to determine the output type
    original_pixel_id = images[0].GetPixelID()
    final_image = sitk.Cast(final_image, original_pixel_id)

# Step 6: Save the combined image
print('saving combined image...')
sitk.WriteImage(final_image, str(args.output))
print('complete...')
print(f'    Output file: {args.output}')