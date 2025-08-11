import ants
import argparse
import numpy as np
from pathlib import Path
import SimpleITK as sitk

"""
This script is a tool to bind multiple Nifti or DICOM images into a single image by resampling each image to a common image space,
as might be necessary for multi-region or whole-body imaging. Overlapping regions are averaged together.
Alan McMillan 2023 (abmcmillan@wisc.edu) 

Inputs:
    --input: The input images/directories to combine.
    --output: The output image.
    --as_float: Convert the images to float before combining.
    --interp_type: Interpolation type to use. Default is linear.
    --voxel_size: The voxel size of the output image (three floating point inputs). Default is to infer from the first input.

Outputs:
    The output image.

Example usage:
    python BindNiftiImages.py --input image1.nii.gz image2.nii.gz --output bound_image.nii.gz
    python BindNiftiImages.py --input dicom_dir1 dicom_dir2 --output bound_image.nii.gz

Notes:
    The current implementation is simple and may not handle oblique images well. Additionally, better strategies for
    merging overlapping regions should be investigated.

Requirements:
    - antspy
    - numpy
    - SimpleITK
    - Python 3.6 or higher
"""

def read_dicom_series(directory, pixeltype='float'):
    """
    Reads a DICOM series from a directory and returns an antspy image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    reader.SetFileNames(dicom_names)
    image_sitk = reader.Execute()

    # Convert SimpleITK image to numpy array
    image_np = sitk.GetArrayFromImage(image_sitk)

    # Get metadata from SimpleITK image
    spacing = image_sitk.GetSpacing()
    origin = image_sitk.GetOrigin()
    direction_sitk = image_sitk.GetDirection()

    # Convert direction matrix to numpy array
    direction_np = np.array(direction_sitk).reshape(len(spacing), len(spacing))

    # Create antspy image from numpy array and metadata
    image_ants = ants.from_numpy(image_np.astype(pixeltype), origin=origin, spacing=spacing, direction=direction_np)

    return image_ants

# 
# 
# The images are combined by resampling each image to a common image space.

# Step 0: Parse the command line arguments
parser = argparse.ArgumentParser(description='Combine multiple Nifti images into a single image.')
parser.add_argument('--input', type=str, nargs='+', help='The input images to combine.', required=True)
parser.add_argument('--output', type=str, help='The output image.', required=True)
parser.add_argument('--as_float', action='store_true', help='Convert the images to float before combining.')
parser.add_argument('--interp_type',
                    choices=['linear', 'nearestNeighbor', 'gaussian', 'bSpline', 'cosineWindowedSinc', 'welchWindowedSinc',
                             'hammingWindowedSinc', 'lanczosWindowedSinc', 'genericLabel'],
                    default='linear', help="Interpolation type to use.")
parser.add_argument('--voxel_size', type=float, nargs=3, help='The voxel size of the output image (three floating point inputs).', required=False)
args = parser.parse_args()

# Check that the input files exist
for fp in args.input:
    path = Path(fp)
    if not (path.is_file() or path.is_dir()):
        raise Exception(f'Input {fp} is not a file or directory.')
    
# Check that the output file does not exist
if Path(args.output).is_file():
    raise Exception(f'Output file {args.output} already exists.')

# Step 1: Load the images
#  Note: we optionally convert the images to float on read
print('loading images...')
if args.as_float:
    pixeltype = 'float'
else:
    pixeltype = None

images = []
for fp in args.input:
    path = Path(fp)
    if path.is_dir():
        print(f"Reading DICOM series from directory: {fp}")
        images.append(read_dicom_series(fp, pixeltype=pixeltype))
    else:
        print(f"Reading Nifti file: {fp}")
        images.append(ants.image_read(fp, pixeltype=pixeltype))


# print some information about the images
real_world_coords = []
for i, img in enumerate(images):
    print( f'file: {args.input[i]}')
    print( f'    shape: {img.shape}' )
    print( f'    spacing: {img.spacing}' )
    print( f'    origin: {img.origin}' )

    # Get the orientation (direction), origin, and spacing
    direction = img.direction
    origin = img.origin
    spacing = img.spacing

    # Initialize a 4x4 matrix with the voxel sizes on the diagonal
    sform = np.eye(4)
    for i in range(3):
        sform[i, i] = spacing[i]

    # Apply the rotation
    sform[0:3, 0:3] = np.dot(sform[0:3, 0:3], np.array(direction).reshape(3,3))

    # Set the translation
    sform[0:3, 3] = origin

    # Multiply by the transformation matrix to get the real-world coordinates
    first_voxel = np.array([0, 0, 0, 1])
    last_voxel = np.array([img.shape[0]-1, img.shape[1]-1, img.shape[2]-1, 1])
    
    first_voxel_real_world = np.dot(sform, first_voxel)[0:3]
    last_voxel_real_world = np.dot(sform, last_voxel)[0:3]
    
    print( f'    [   0,   0,   0] in mm: {first_voxel_real_world}' )
    print( f'    [maxX,maxY,maxZ] in mm: {last_voxel_real_world}' )

    real_world_coords.append( first_voxel_real_world )
    real_world_coords.append( last_voxel_real_world )

# Step 2: Determine the spatial extent in world space coordinates
print('computing spatial extents...')
min_coords = np.min( real_world_coords, axis=0 )
max_coords = np.max( real_world_coords, axis=0 )
extent_range = max_coords - min_coords

print( f'    min_extent: {min_coords}' )
print( f'    max_extent: {max_coords}' )
print( f'    extent_range: {extent_range}')

# Determine the voxel size of the new image
if args.voxel_size is None:
    # infer the voxel size from the first input
    newimg_spacing = images[0].spacing
    newimg_spacing = tuple( value/2 for value in newimg_spacing ) # cut the pixel size in half
    print( f'    new image spacing (inferred from first input): {newimg_spacing}' )
else:
    # use the voxel size provided by the user
    newimg_spacing = tuple( args.voxel_size )
# Compute size of the new image, and make sure to round up to ensure all images fit
newimg_size = tuple( np.ceil(extent_range / newimg_spacing).astype(int) )
# determine the origin of the new image from extent_range and sform
# TODO - consider better strategies for the output origin and direction, there might be troubles with oblique images
newimg_origin = tuple( min_coords ) # use the minimum coordinates as the origin
# determine the direction of the new image
#newimg_direction = images[0].direction
newimg_direction = np.eye(3) # use identity matrix as the direction
# determine the pixeltype of the new image
newimg_pixeltype = images[0].pixeltype # note that if the as_float parameter was passed, the images are already float

# Step 3: Create new empty matrix
print('creating new image...')
newimg = ants.make_image( imagesize=newimg_size, spacing=newimg_spacing, origin=newimg_origin, direction=newimg_direction, pixeltype=newimg_pixeltype )

print( 'bound image' )
print( f'    shape: {newimg.shape}' )
print( f'    spacing: {newimg.spacing}' )
print( f'    origin: {newimg.origin}' )
print( f'    min_extent: {newimg_origin}' )
print( f'    max_extent: {newimg.origin + newimg.spacing * np.array(newimg.shape)}' )
print( f'    pixeltype: {newimg_pixeltype}' )

print('binding images...')
# Step 4: Resample each image to the new image
# create an image to keep track of the overlap
overlap_img = ants.image_clone( newimg )
# loop through each image and resample it to the new image
# TODO - consider better strategies for combining images verses simple averaging
for i, img in enumerate(images):
    # calculate overlap by resampling a binary image
    curr_overlap_img = ants.make_image( imagesize=img.shape, voxval=1, spacing=img.spacing, origin=img.origin, direction=img.direction, pixeltype=img.pixeltype )
    transformed_overlap_img = ants.resample_image_to_target( image=curr_overlap_img, target=newimg, interp_type=args.interp_type )
    overlap_img += transformed_overlap_img
    
    transformed_img = ants.resample_image_to_target( image=img, target=newimg, interp_type=args.interp_type )
    newimg += transformed_img

# Step 5: Intensity correct the overlapping regions
newimg /= overlap_img

# Step 6: Save the combined image
ants.image_write(newimg, str(args.output) )
print('complete...')
print( f'    Output file: {args.output}' )