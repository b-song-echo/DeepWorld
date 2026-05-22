# Documentation

## ScanNet++ Toolbox

Check out our [ScanNet++ Toolbox](https://github.com/scannetpp/scannetpp) in github. It provides tools and code to

-   Read the dataset structure
-   Decode iPhone RGB, depth, and mask video
-   Undistort DSLR fisheye images into pinhole
-   Render high-res depth maps from the mesh for DSLR and iPhone frames
-   [Nerfstudio](https://docs.nerf.studio/) dataparser for DSLR images is ready (see this [PR](https://github.com/nerfstudio-project/nerfstudio/pull/2498))
-   Prepare training data for semantic tasks
-   Official evaluation code for the benchmark

## 3D Gaussian Splatting on ScanNet++

Check out our [3DGS example](https://github.com/scannetpp/3DGS-demo) in github. It provides tools and code to visualize the frames and train 3D Gaussian Splatting on the scenes in ScanNet++. It also includes scripts to render images for submitting the NVS benchmark (DSLR).

## Data Structure

The ScanNet++ dataset currently consists of 1006 scenes. The default download (low-res DSLR images, iPhone data, 3D meshes and semantics) occupies about 1.5 TB on disk.

Asset group

Download size

DSLR (2 MP), iPhone, Meshes and semantics (default download)

1.5 TB

DSLR (2 MP)

371 GB

DSLR 2MP + 33MP (hi-res)

9 TB

Meshes and semantics

132 GB

Point clouds

720 GB

Panocam

319 GB

The data download contains one folder per scene containing laser scan, DSLR and iPhone data, and several metadata files. The data is organized as follows:

-   **split/**
    -   **nvs\_sem\_train.txt**: Training set for NVS and semantic tasks with 856 scenes
    -   **nvs\_sem\_val.txt**: Validation set for NVS and semantic tasks with 50 scenes
    -   **nvs\_test.txt**: Test set for NVS with 50 scenes (no scan data)
    
    -   **nvs\_test\_small.txt**: Smaller test set for NVS with 12 scenes, which is a subset of **nvs\_test.txt** (no scan data)
    
    -   **nvs\_test\_iphone.txt**: Test set for NVS with iPhone data with 12 scenes (no scan data)
    -   **sem\_test.txt**: Test set for semantic tasks with 50 scenes
    -   Each file contains lists of Scene IDs in the respective split
-   **metadata/**
    -   **scene\_types.json**: scene ID to scene type mapping for all scenes
    -   **semantic\_classes.txt**: list of semantic classes
    -   **instance\_classes.txt**: subset of semantic classes that have instances (i.e., excludes wall, ceiling, floor, ..)
    -   **semantic\_benchmark/**
        -   **top100.txt**: top 100 semantic classes for semantic segmentation benchmark
        -   **top100\_instance.txt**: subset of 100 semantic classes for instance segmentation benchmark
        -   **map\_benchmark.csv**: mapping from raw semantic labels to benchmark labels
-   **data/<scene\_id>/**
    -   **scans/**
        
        -   **pc\_aligned.ply**: point cloud from laser scanner, axis-aligned
        -   **pc\_aligned\_mask.txt**: indices of anonymized points
        -   **scanner\_poses.json**: contains scanner positions, 4x4 transformation matrix for each position
        -   **mesh\_aligned\_0.05.ply**: mesh decimated to 5% size, obtained from point cloud
        -   **mesh\_aligned\_0.05\_mask.txt**: indices of mesh vertices with anonymization applied
        -   **mesh\_aligned\_0.05\_semantic.ply**
            -   The vertex “label” property contains the integer semantic label into the classes in semantic\_classes.txt
            -   Unlabeled vertices have the label -100
        -   **segments.json**: json\_data\[“segIndices”\] contains the segment ID for each vertex
        -   **segments\_anno.json:**
            -   json\_data\[i\] corresponds to a single annotated object
                -   **“label”**: the semantic label of this object
                -   **“segments”**: all the segments belonging to this object
        
    -   **dslr/**
        -   **resized\_images**: Fisheye DSLR images, resized, JPG
        -   **resized\_anon\_masks**: PNG. Specifies the pixels that have been anonymized (0: invalid, 255: valid pixels).
        -   **original\_images**: Full resolution images, JPG
        -   **original\_anon\_masks**: PNG. Similar to resized masks
        -   **resized\_undistorted\_images**: Undistorted DSLR images with the same resolution as the resized images, JPG
        -   **resized\_undistorted\_masks**: PNG. Similar to resized masks
        -   **colmap**: contains the colmap camera model that has been aligned with the 3D scans, which implies the poses are in metric scale. Make sure to use this if you want to do 2D-3D matching between the provided mesh.
            -   **cameras.txt**: Contain the camera type (OPENCV\_FISHEYE) and the intrinsic parameters (fx, fy, cx, cy, distortion parameters)
            -   **images.txt**: Contain extrinsics of each image: qvec (quaternion) and tvec
            -   **points3D.txt**: Contain 3D feature points used by COLMAP
            -   Useful references:
                -   [Colmap docs for camera model](https://colmap.github.io/cameras.html)
                -   [More info in colmap source code](https://github.com/colmap/colmap/blob/5c1d58a085920c41cf2d9c892da8041b1fc0c86d/src/colmap/sensor/models.h#L249)
                -   [Python reader provided by colmap](https://github.com/colmap/colmap/blob/5c1d58a085920c41cf2d9c892da8041b1fc0c86d/scripts/python/read_write_model.py)
            -   Python (Open3D) [visualizer](https://github.com/colmap/colmap/blob/5c1d58a085920c41cf2d9c892da8041b1fc0c86d/scripts/python/visualize_model.py) provided by COLMAP
        -   **nerfstudio/**
            -   **transforms.json**
                -   Contains the same camera poses in the format used by Nerfstudio, [OpenGL/Blender convention](https://docs.nerf.studio/quickstart/data_conventions.html). The coordinate system is different from OpenCV/COLMAP convention.
                -   **poses:**
                    -   **frames, test\_frames**: contain poses for train and test images respectively
                -   **mask**: filename of binary mask file
                -   **is\_bad**: indicates if the image is blurry or contains heavy shadows.
                -   Camera model (as above):
                    -   contained in fl\_x, fl\_y, .., k1, k2, k3, k4, camera\_model
                    -   The intrinsics are corresponds to the resized images
                -   **has\_mask**: global flag for the scene, indicating if it has anonymized masks or not
            -   **transforms\_undistorted.json**: similar to transforms.json but the undistorted DSLR version
        -   **train\_test\_lists.json**
            -   **json\[“train”\]**: training images
            -   **json\[“test”\]**: novel views, test images
            -   The split here is the same as the one in nerfstudio/transform.json
            -   **json\[“has\_masks”\]**: global flag for the scene, indicating if it has anonymized masks or not
    -   **iphone/**
        -   **rgb.mkv**: full RGB video, 60 FPS
        -   **rgb\_mask.mkv**: Video of anonymization masks, lossless compression. After decode, it's similar to the masks in DSLR.
        -   **depth.bin**: Depth images as 16 bit png in millimeters in a single binary file from iPhone Lidar sensor. The depth images are aligned with the RGB images.
        -   **rgb**: RGB frames from the video, subsampled. Obtained by running processing script on rgb.mkv. The resolution is 1920 x 1440.
        -   **depth**: Depth images as 16 bit png in millimeters. Obtained by running processing script on depth.bin. The depth images are aligned with the RGB images but with much lower resolution: 256 x 192.
        -   **pose\_intrinsic\_imu.json**: contains ARKit poses and IMU information from the iPhone
            -   json\["poses"\] contain a 4x4 camera-to-world extrinsic matrix from raw ARKit output. The coordinate system is right-handed. +Z is the camera direction.
            -   json\["intrinsic"\] contains a 3x3 intrinsic matrix of the RGB image
            -   json\["aligned\_poses"\] contains ARKit poses that are scaled and transformed to our mesh space
            -   There are no intrinsics for Lidar depth provided by the iPhone. The user can scale the RGB intrinsic for the Lidar depth map since RGB and depth are aligned.
        -   **nerfstudio**: similar to DSLR
        -   **colmap:** similar to DSLR. Images here have been filtered based on agreement of depth between iPhone Lidar and the laser scanner. The camera model is OPENCV, which contains 4 distortion parameters: k1, k2 p1, p2.
        -   **exif.json:** EXIF information for each frame in the video
    -   **panocam/**
        -   **images:** <scan\_id>.jpg with aspect ratio approximately 2:1 or 2.5:1. Corresponds to the scan pose **i** in **scans/scanner\_poses.json**
        -   **anon\_mask:** <scan\_id>.png similar to DSLR anonymization masks
        -   **depth:** <scan\_id>.png depth images in millimeters in 16 bit PNG format
        -   **azim:** <scan\_id>.png azimuth angle images, radians\*1000 in 16 bit PNG format
        -   **elev:** <scan\_id>.png elevation angle images, radians\*1000 in 16 bit PNG format
        -   Additionally, **resized\_\*** folders for resized images, depth, mask, azimuth, elevation which are resized to 1/4 of the original size.
        -   See [example code](https://github.com/scannetpp/scannetpp?tab=readme-ov-file#backproject-panocam-images) for usage.
-   All data is anonymized using the magenta color with RGB value (255,0,255). The user may fill those regions with any color using the given binary mask.
-   To ensure fair comparison, the scenes in nvs\_test split do **not** contain 3D information like meshes and iphone depth maps.