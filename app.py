from functools import reduce

import math
import os
import uuid
from enum import Enum
from typing import List, Optional
import numpy as np
import xarray as xr
from colorthief import ColorThief
from matplotlib import cm
from PIL import Image
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mean
import cv2
from mikro.api.schema import (
    MetricFragment,
    OmeroFileFragment,
    RepresentationFragment,
    RepresentationVariety,
    ROIFragment,
    ROIType,
    ThumbnailFragment,
    RoiTypeInput,
    DatasetFragment,
    StageFragment,
    PositionFragment,
    create_position,
    create_metric,
    create_roi,
    create_stage,
    create_thumbnail,
    create_channel,
    from_xarray,
    get_representation,
    OmeroRepresentationInput,
    PhysicalSizeInput,
    AffineMatrix,
    RepresentationViewInput,
    OmeroRepresentationInput,
    PhysicalSizeInput,
    InputVector,
    create_era,
    EraFragment,
    TableFragment,
    from_df,
)
import operator
from arkitekt import register, log, group
from functools import partial
from skimage import transform
from api.mikro import get_filedataset
import datetime
from typing import Tuple
from skimage import data
import skimage
from scipy import ndimage
from sklearn.cluster import DBSCAN
import pandas as pd

class Colormap(Enum):
    VIRIDIS = partial(cm.viridis)  # partial needed to make it register as an enum value
    PLASMA = partial(cm.plasma)


@register(collections=["conversion", "rendering"])
def array_to_image(
    rep: RepresentationFragment,
    rescale=True,
    max=True,
    cm: Colormap = Colormap.VIRIDIS,
) -> ThumbnailFragment:
    """Thumbnail Image

    Generates Thumbnail for the Image

    Args:
        rep (Representation): The to be converted Image
        rescale (bool, optional): SHould we rescale the image to fit its dynamic range?. Defaults to True.
        max (bool, optional): Automatically z-project if stack. Defaults to True.
        cm: (Colormap, optional): The colormap to use. Defaults to Colormap.VIRIDIS.

    Returns:
        Thumbnail: The Thumbnail
    """

    array = rep.data

    cm = cm.value

    if "z" in array.dims:
        if not max:
            raise Exception("Set Max to Z true if you want to convert image stacks")
        array = array.max(dim="z")

    if "t" in array.dims:
        array = array.sel(t=0)

    if "c" in array.dims:
        # Check if we have to convert to monoimage
        if array.c.size == 1:
            array = array.sel(c=0)

            if rescale is True:
                log("Rescaling")
                min, max = array.min(), array.max()
                image = np.interp(array, (min, max), (0, 255)).astype(np.uint8)
            else:
                image = (array * 255).astype(np.uint8)

            print(cm)
            mapped = cm(image)

            finalarray = (mapped * 255).astype(np.uint8)

        else:
            if array.c.size >= 3:
                array = array.sel(c=[0, 1, 2]).transpose(*list("yxc")).data
            elif array.c.size == 2:
                # Two Channel Image will be displayed with a Dark Channel
                array = np.concatenate(
                    [
                        array.sel(c=[0, 1]).transpose(*list("yxc")).data,
                        np.zeros((array.x.size, array.y.size, 1)),
                    ],
                    axis=2,
                )

            if rescale is True:
                log("Rescaling")
                min, max = array.min(), array.max()
                finalarray = np.interp(array.compute(), (min, max), (0, 255)).astype(
                    np.uint8
                )
            else:
                finalarray = (array.compute() * 255).astype(np.uint8)

    else:
        raise NotImplementedError("Image Does not provide the channel Argument")

    print("Final Array Shape", finalarray.shape)

    temp_file = uuid.uuid4().hex + ".jpg"

    aspect = finalarray.shape[0] / finalarray.shape[1]

    img = Image.fromarray(finalarray)
    img = img.convert("RGB")

    if finalarray.shape[0] > 512:
        img = img.resize((512, int(512 * aspect)), Image.Resampling.BILINEAR)
    img.save(temp_file, quality=80)

    ColorThief(temp_file)

    # with open(temp_file, "rb") as image_file:
    #     hash = blurhash.encode(image_file, x_components=6, y_components=6)
    #     print(hash)

    hash = None
    print("Hash", hash)

    major_color = None

    # major_color = "#%02x%02x%02x" % color_thief.get_color(quality=3)
    # print("Major Color", major_color)

    th = create_thumbnail(
        file=open(temp_file, "rb"),
        rep=rep,
        major_color=major_color,
        blurhash=hash,
    )
    print("Done")
    os.remove(temp_file)
    return th


@register(collections=["quantitative"])
def measure_max(
    rep: RepresentationFragment,
    key: str = "max",
) -> MetricFragment:
    """Measure Max

    Measures the maxium value of an image

    Args:
        rep (OmeroFiRepresentationFragmentle): The image
        key (str, optional): The key to use for the metric. Defaults to "max".

    Returns:
        Representation: The Back
    """
    return create_metric(
        key=key, value=float(rep.data.max().compute()), representation=rep
    )


@register(collections=["creation"])
def create_era_func(
    name: str = "max",
) -> EraFragment:
    """Create Era Now

    Creates an era with the current time as a starttime

    Returns:
        Representation: The Back
    """
    return create_era(name=name, start=datetime.datetime.now())


@register(collections=["collection"])
def iterate_images(
    dataset: DatasetFragment,
) -> RepresentationFragment:
    """Iterate Images

    Iterate over all images in a dataset

    Args:
        rep (Dataset): The dataset

    yields:
        Representation: The image
    """
    for x in get_filedataset(dataset).representations:
        yield x


@register(collections=["quantitative"])
def measure_sum(
    rep: RepresentationFragment,
    key: str = "Sum",
) -> MetricFragment:
    """Measure Sum

    Measures the sum of all values of an image

    Args:
        rep (OmeroFiRepresentationFragmentle): The image
        key (str, optional): The key to use for the metric. Defaults to "max".

    Returns:
        Representation: The Back
    """
    return create_metric(
        key=key, value=float(rep.data.sum().compute()), representation=rep
    )


@register(collections=["quantitative"])
def measure_fraction(
    rep: RepresentationFragment,
    key: str = "Fraction",
    value: float = 1,
) -> MetricFragment:
    """Measure Fraction

    Measures the appearance of this value in the image (0-1)

    Args:
        rep (OmeroFiRepresentationFragmentle): The image
        key (str, optional): The key to use for the metric. Defaults to "max".

    Returns:
        Representation: The Back
    """
    x = rep.data == value
    sum = x.sum().compute()
    all_values = reduce(lambda x, t: x * t, rep.data.shape, 1)

    return create_metric(key=key, value=float(sum / all_values), representation=rep)


@register(collections=["quantitative"])
def measure_basics(
    rep: RepresentationFragment,
) -> List[MetricFragment]:
    """Measure Basic Metrics

    Measures basic meffffftrics of an image like max, mifffffn, mean

    Args:
        rep (OmeroFiRepresentationFragmentle): The image

    Returns:
        Representation: The Back
    """

    x = rep.data.compute()

    return [
        create_metric(key="maximum", value=float(x.max()), representation=rep),
        create_metric(key="mean", value=float(x.mean()), representation=rep),
        create_metric(key="min", value=float(x.min()), representation=rep),
    ]


@register(collections=["conversion"])
def t_to_frame(
    rep: RepresentationFragment,
    interval: int = 1,
    key: str = "frame",
) -> ROIFragment:
    """T to Frame

    Converts a time series to a single frame

    Args:
        rep (RepresentationFragment): The Representation
        frame (int): The frame to select

    Returns:
        RepresentationFragment: The new Representation
    """
    assert "t" in rep.data.dims, "Cannot convert non time series to frame"

    for i in range(rep.data.sizes["t"]):
        if i % interval == 0:
            yield create_roi(
                representation=rep,
                label=f"{key} {i}",
                type=RoiTypeInput.FRAME,
                tags=[f"t{i}", "frame"],
                vectors=[InputVector(t=i), InputVector(t=i + interval)],
            )


@register(collections=["conversion"])
def z_to_slice(
    rep: RepresentationFragment,
    interval: int = 1,
    key: str = "Slice",
) -> ROIFragment:
    """Z to Slice

    Creates a slice roi for each z slice

    Args:
        rep (RepresentationFragment): The Representation
        frame (int): The frame to select

    Returns:
        RepresentationFragment: The new Representation
    """
    assert "z" in rep.data.dims, "Cannot convert non time series to frame"

    for i in range(rep.data.sizes["z"]):
        if i % interval == 0:
            yield create_roi(
                representation=rep,
                label=f"{key} {i}",
                type=RoiTypeInput.SLICE,
                tags=[f"z{i}", "frame"],
                vectors=[InputVector(z=i), InputVector(z=i + interval)],
            )


def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


@register(collections=["conversion", "cropping"])
def crop_image(
    roi: ROIFragment, rep: Optional[RepresentationFragment]
) -> RepresentationFragment:
    """Crop Image

    Crops an Image based on a ROI

    Args:
        roi (ROIFragment): The Omero File
        rep (Optional[RepresentationFragment], optional): The Representation to be cropped. Defaults to the one of the ROI.

    Returns:
        Representation: The Back
    """
    if rep is None:
        rep = get_representation(roi.representation.id)

    array = rep.data
    if roi.type == ROIType.RECTANGLE:
        x_start = roi.vectors[0].x
        y_start = roi.vectors[0].y
        x_end = roi.vectors[0].x
        y_end = roi.vectors[0].y

        for vector in roi.vectors:
            if vector.x < x_start:
                x_start = vector.x
            if vector.x > x_end:
                x_end = vector.x
            if vector.y < y_start:
                y_start = vector.y
            if vector.y > y_end:
                y_end = vector.y

        roi.vectors[0]

        array = array.sel(
            x=slice(math.floor(x_start), math.floor(x_end)),
            y=slice(math.floor(y_start), math.floor(y_end)),
        )

        return from_xarray(
            array,
            name="Cropped " + rep.name,
            tags=["cropped"],
            origins=[rep],
            roi_origins=[roi],
        )

    if roi.type == ROIType.FRAME:
        array = array.sel(
            t=slice(math.floor(roi.vectors[0].t), math.floor(roi.vectors[1].t))
        )

        return from_xarray(
            array,
            name="Cropped " + rep.name,
            tags=["cropped"],
            origins=[rep],
            roi_origins=[roi],
        )

    if roi.type == ROIType.SLICE:
        array = array.sel(
            z=slice(math.floor(roi.vectors[0].z), math.floor(roi.vectors[1].z))
        )

        return from_xarray(
            array,
            name="Cropped " + rep.name,
            tags=["cropped"],
            origins=[rep],
            roi_origins=[roi],
        )

    raise Exception(f"Roi Type {roi.type} not supported")


class DownScaleMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"


@register(collections=["processing", "scaling"])
def downscale_image(
    rep: RepresentationFragment,
    factor: int = 2,
    depth=0,
    method: DownScaleMethod = DownScaleMethod.MEAN,
) -> RepresentationFragment:
    """Downscale

    Scales down the Representatoi by the factor of the provided

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    s = tuple([1 if c == 1 else factor for c in rep.data.squeeze().shape])

    newrep = multiscale(rep.data.squeeze(), windowed_mean, s)

    return from_xarray(
        newrep[1],
        name=f"Downscaled {rep.name} by {factor}",
        tags=[f"scale-{factor}"],
        variety=RepresentationVariety.VOXEL,
        origins=[rep],
    )


@register(collections=["processing", "scaling"])
def rescale(
    rep: RepresentationFragment,
    factor_x: float = 2.0,
    factor_y: float = 2.0,
    factor_z: float = 2.0,
    factor_t: float = 1.0,
    factor_c: float = 1.0,
    anti_alias: bool = True,
    method: DownScaleMethod = DownScaleMethod.MEAN,
) -> RepresentationFragment:
    """Rescale

    Rescale the dimensions by the factors provided

    Args:
        rep (RepresentationFragment): The Image we should rescale

    Returns:
        RepresentationFragment: The Rescaled image
    """

    scale_map = {
        "x": factor_x,
        "y": factor_y,
        "z": factor_z,
        "t": factor_t,
        "c": factor_c,
    }

    squeezed_data = rep.data.squeeze()
    dims = squeezed_data.dims

    s = tuple([scale_map[d] for d in dims])

    newrep = transform.rescale(squeezed_data.data, s, anti_aliasing=anti_alias)

    return from_xarray(
        xr.DataArray(newrep, dims=dims),
        name=f"Rescaled {rep.name}",
        tags=[f"scale-{key}-{factor}" for key, factor in scale_map.items()],
        variety=RepresentationVariety.VOXEL,
        origins=[rep],
    )


@register(collections=["processing", "scaling"])
def resize(
    rep: RepresentationFragment,
    dim_x: Optional[int],
    dim_y: Optional[int],
    dim_z: Optional[int],
    dim_t: Optional[int],
    dim_c: Optional[int],
    anti_alias: bool = True,
) -> RepresentationFragment:
    """Resize

    Resize the image to the dimensions provided

    Args:
        rep (RepresentationFragment): The Image we should resized

    Returns:
        RepresentationFragment: The resized image
    """

    scale_map = {
        "x": dim_x or rep.data.sizes["x"],
        "y": dim_y or rep.data.sizes["y"],
        "z": dim_z or rep.data.sizes["z"],
        "t": dim_t or rep.data.sizes["t"],
        "c": dim_c or rep.data.sizes["c"],
    }

    squeezed_data = rep.data.squeeze()
    dims = squeezed_data.dims

    s = tuple([scale_map[d] for d in dims])

    newrep = transform.resize(
        squeezed_data.data, s, anti_aliasing=anti_alias, preserve_range=True
    )

    newrep = skimage.util.img_as_uint(newrep)

    return from_xarray(
        xr.DataArray(newrep, dims=dims),
        name=f"Resized {rep.name}",
        tags=[f"resize-{key}-{factor}" for key, factor in scale_map.items()],
        variety=RepresentationVariety.VOXEL,
        origins=[rep],
    )


class CropMethod(Enum):
    CENTER = "mean"
    TOP_LEFT = "top-left"
    BOTTOM_RIGHT = "bottom-right"


class ExpandMethod(Enum):
    PAD_ZEROS = "zeros"


@register(
    groups={
        "ensure_dim_x": ["advanced"],
        "ensure_dim_y": ["advanced"],
        "ensure_dim_z": ["advanced"],
        "crop_method": ["advanced"],
        "pad_method": ["advanced"],
        "anti_alias": ["advanced"],
    },
    port_groups=[group(key="advanced", hidden=True)],
    collections=["processing", "scaling"],
)
def resize_to_physical(
    rep: RepresentationFragment,
    rescale_x: Optional[float],
    rescale_y: Optional[float],
    rescale_z: Optional[float],
    ensure_dim_x: Optional[int],
    ensure_dim_y: Optional[int],
    ensure_dim_z: Optional[int],
    crop_method: CropMethod = CropMethod.CENTER,
    pad_method: ExpandMethod = ExpandMethod.PAD_ZEROS,
    anti_alias: bool = True,
) -> RepresentationFragment:
    """Resize to Physical

    Resize the image to match the physical size of the dimensions,
    if the physical size is not provided, it will be assumed to be 1.

    Additional dimensions will be cropped or padded according to the
    crop_method and pad_method if the ensure_dim is provided

    Args:
        rep (RepresentationFragment): The Image we should resized
        rescale_x (Optional[float]): The physical size of the x dimension
        rescale_y (Optional[float]): The physical size of the y dimension
        rescale_z (Optional[float]): The physical size of the z dimension
        ensure_dim_x (Optional[int]): The size of the x dimension
        ensure_dim_y (Optional[int]): The size of the y dimension
        ensure_dim_z (Optional[int]): The size of the z dimension
        crop_method (CropMethod, optional): The method to crop the image. Defaults to crop center.
        pad_method (ExpandMethod, optional): The method to pad the image. Defaults to expand with zeros.

    Returns:
        RepresentationFragment: The resized image
    """
    if not rep.omero or not rep.omero.physical_size:
        raise ValueError("Input Image has no physical size provided")

    originial_scale = rep.omero.physical_size
    scale_map = {
        "x": rescale_x / rep.omero.physical_size.x if rescale_x else 1,
        "y": rescale_y / rep.omero.physical_size.y if rescale_y else 1,
        "z": rescale_z / rep.omero.physical_size.z if rescale_z else 1,
        "t": 1,
        "c": 1,
    }

    squeezed_data = rep.data.squeeze()
    dims = squeezed_data.dims

    s = tuple([scale_map[d] for d in dims])

    if not all([d == 1 for d in s]):
        newrep = transform.rescale(
            squeezed_data.data.compute(), s, anti_aliasing=anti_alias, preserve_range=True
        )
    else:
        newrep = squeezed_data.data.compute()


    new_array = xr.DataArray(newrep, dims=dims)

    if ensure_dim_x or ensure_dim_y or ensure_dim_z:
        print(newrep.shape)

        size_map = {
            "x": ensure_dim_x or newrep.shape[2],
            "y": ensure_dim_y or newrep.shape[1],
            "z": ensure_dim_z or newrep.shape[0],
            "t": rep.data.sizes["t"],
            "c": rep.data.sizes["c"],
        }

        s = tuple([size_map[d] for d in dims])
        new_array = cropND(
            new_array,
            s,
        )

    return from_xarray(
        new_array,
        name=f"Resized {rep.name}",
        tags=[f"resize-{key}-{factor}" for key, factor in scale_map.items()],
        variety=RepresentationVariety.VOXEL,
        omero=OmeroRepresentationInput(
            physicalSize=PhysicalSizeInput(
                x=rescale_x if rescale_x else originial_scale.x,
                y=rescale_y if rescale_y else originial_scale.y,
                z=rescale_z if rescale_z else originial_scale.z,
            ),
        ),
        origins=[rep],
    )


class ThresholdMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


@register(collections=["processing", "thresholding"])
def threshold_image(
    rep: RepresentationFragment,
    threshold: float = 0.5,
    method: ThresholdMethod = ThresholdMethod.MEAN,
) -> RepresentationFragment:
    """Binarize

    Binarizes the image based on a threshold

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    print(method)
    if method == ThresholdMethod.MEAN.value:
        m = rep.data.mean()
    if method == ThresholdMethod.MAX.value:
        m = rep.data.max()
    if method == ThresholdMethod.MIN.value:
        m = rep.data.min()

    newrep = rep.data > threshold * m

    return from_xarray(
        newrep,
        name=f"Thresholded {rep.name} by {threshold} through {method}",
        tags=[f"thresholded-{threshold}", f"{method}-binarized"],
        variety=RepresentationVariety.VOXEL,
        origins=[rep],
    )


@register(collections=["processing", "projection"])
def maximum_intensity_projection(
    rep: RepresentationFragment,
) -> RepresentationFragment:
    """Maximum Intensity Projection

    Projects the image onto the maximum intensity along the z axis

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    m = rep.data.max(dim="z").compute()

    print("Maximum?")
    print(m.max())
    print(m.min())

    return from_xarray(
        m,
        name=f"Maximum Intensity Projection of {rep.name}",
        tags=["maximum-intensity-projection"],
        variety=RepresentationVariety.VOXEL,
        origins=[rep],
    )


class CV2NormTypes(Enum):
    NORM_INF = cv2.NORM_INF
    NORM_L1 = cv2.NORM_L1
    NORM_L2 = cv2.NORM_L2
    NORM_MINMAX = cv2.NORM_MINMAX
    NORM_RELATIVE = cv2.NORM_RELATIVE
    NORM_TYPE_MASK = cv2.NORM_TYPE_MASK


@register(collections=["processing", "thresholding", "adaptive"])
def adaptive_threshold_image(
    rep: RepresentationFragment,
    normtype: CV2NormTypes = CV2NormTypes.NORM_MINMAX,
) -> RepresentationFragment:
    """Adaptive Binarize

    Binarizes the image based on a threshold

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """
    x = rep.data.compute()

    thresholded = xr.DataArray(np.zeros_like(x), dims=x.dims, coords=x.coords)

    for c in range(x.sizes["c"]):
        for z in range(x.sizes["z"]):
            for t in range(x.sizes["t"]):
                img = x.sel(c=c, z=z, t=t)
                normed = cv2.normalize(img.data, None, 0, 255, normtype, cv2.CV_8U)
                thresholded[c, t, z, :, :] = cv2.adaptiveThreshold(
                    normed,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )

    return from_xarray(
        thresholded,
        name=f"Adaptive Thresholded {rep.name}",
        tags=["adaptive-thresholded"],
        variety=RepresentationVariety.VOXEL,
        origins=[rep],
    )


class CV2NormTypes(Enum):
    NORM_INF = cv2.NORM_INF
    NORM_L1 = cv2.NORM_L1
    NORM_L2 = cv2.NORM_L2
    NORM_MINMAX = cv2.NORM_MINMAX
    NORM_RELATIVE = cv2.NORM_RELATIVE
    NORM_TYPE_MASK = cv2.NORM_TYPE_MASK


@register(collections=["processing", "thresholding", "adaptive"])
def otsu_thresholding(
    rep: RepresentationFragment,
    gaussian_blur: bool = False,
) -> RepresentationFragment:
    """Otsu Thresholding

    Binarizes the image based on a threshold, using Otsu's method
    with option to guassian blur the image before, images are normalized
    for each x,y slice before thresholding.

    Args:
        rep (RepresentationFragment): The Image to be thresholded
        gaussian_blur (bool): Whether to apply a gaussian blur before thresholding (kernel_size=5)

    Returns:
        RepresentationFragment: The thresholded image
    """
    x = rep.data.compute()

    thresholded = xr.DataArray(np.zeros_like(x, dtype=np.uint8), dims=x.dims, coords=x.coords)
    print("Hallo")
    print(x.min())
    print(x.max())

    for c in range(x.sizes["c"]):
        for z in range(x.sizes["z"]):
            for t in range(x.sizes["t"]):
                img = x.sel(c=c, z=z, t=t).data
                 # Find min and max values
                min_val = np.min(img)
                max_val = np.max(img)
                
                # Apply min-max normalization
                normalized_arr = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                if gaussian_blur:
                    normalized_arr = cv2.GaussianBlur(img, (5, 5), 0)
                print(img)
                threshold, image = cv2.threshold(
                    normalized_arr,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                print(image, threshold)
                print(image.dtype)
                thresholded[c, t, z, :, :] = image

    print(thresholded.dtype)

    return from_xarray(
        thresholded,
        name=f"Otsu Thresholded {rep.name}",
        tags=["otsu-thresholded"],
        variety=RepresentationVariety.MASK,
        origins=[rep],
    )


@register(collections=["conversion"])
def roi_to_position(
    roi: ROIFragment,
) -> PositionFragment:
    """Roi to Position

    Transforms a ROI into a Position on the governing stage

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        RepresentationFragment: The Downscaled image
    """

    smart_image = get_representation(roi.representation)

    while smart_image.omero is None or smart_image.omero.positions is None:
        smart_image = get_representation(smart_image.origins[0])
        assert (
            smart_image.shape == roi.representation.shape
        ), "Could not find a matching position is not in the same space as the original (probably through a downsampling, or cropping)"

    omero = smart_image.omero
    affine_transformation = omero.affine_transformation
    shape = smart_image.shape
    position = smart_image.omero.positions[0]

    # calculate offset between center of roi and center of image
    print(position)
    print(roi.get_vector_data(dims="ctzyx"))
    center = roi.center_as_array()
    print(center)

    image_center = np.array(shape) / 2
    print(image_center[2:])
    print(center[2:])
    offsetz, offsety, offsetx = image_center[2:]
    z, y, x = center[2:]

    x_from_center = x - offsetx
    y_from_center = y - offsety
    z_from_center = z - offsetz

    # TODO: check if this is correct and extend to 3d
    vec_center = np.array([x_from_center, y_from_center, z_from_center])
    vec = np.matmul(np.array(affine_transformation).reshape((3, 3)), vec_center)
    new_pos_x, new_pos_y, new_pos_z = (
        np.array([position.x, position.y, position.z]) + vec
    )

    print(vec)

    print("Affine", affine_transformation)

    return create_position(
        stage=position.stage,
        name=f"Position of {roi.label or 'Unknown ROI'}",
        x=new_pos_x,
        y=new_pos_y,
        z=new_pos_z,
        roi_origins=[roi],
    )


@register(collections=["conversion"])
def roi_to_physical_dimensions(
    roi: ROIFragment,
) -> Tuple[float, float]:
    """Rectangular Roi to Dimensions

    Measures the size of a Rectangular Roi in microns
    (only works for Rectangular ROIS)

    Parameters
    ----------
    roi : ROIFragment
        The roi to measure

    Returns
    -------
    height: float
        The height of the ROI in microns
    width: float
        The width of the ROI in microns
    """
    assert roi.type == ROIType.RECTANGLE, "Only works for rectangular ROIs"
    smart_image = get_representation(roi.representation)

    while smart_image.omero is None or smart_image.omero.physical_size is None:
        smart_image = get_representation(smart_image.origins[0])
        assert (
            smart_image.shape == roi.representation.shape
        ), "Could not find a matching position is not in the same space as the original (probably through a downsampling, or cropping)"

    physical_size = smart_image.omero.physical_size

    # Convert to a numpy array for easier manipulation
    points = roi.get_vector_data(dims="yx")

    # Find the minimum and maximum x and y coordinates
    min_y, min_x = np.min(points, axis=0)
    max_y, max_x = np.max(points, axis=0)

    # Calculate the width and height
    width = max_x - min_x
    height = max_y - min_y

    return width * physical_size.x, height * physical_size.y


@register(collections=["conversion"])
def rois_to_positions(
    roi: List[ROIFragment],
) -> List[PositionFragment]:
    """Rois to Positions

    Transforms a List of Rois into a List of Positions on the governing stage

    Args:
        rep (RepresentationFragment): The Image where we should count cells

    Returns:
        List[PositionFragment]: The Downscaled image
    """
    positions = []
    for r in roi:
        positions.append(roi_to_position(r))

    return positions


@register(collections=["creation"])
def create_stage_from_name(
    name: str,
) -> StageFragment:
    """Create New Stage

    Creates a new stage with the given name

    """

    return create_stage(name=name)


@register(collections=["conversion"])
def merge_positions_to_stage(
    name: List[PositionFragment],
) -> StageFragment:
    """Merge positions to stage

    Creates a new stage with the given name

    """
    s = create_stage(name="jahh")

    for i in name:
        create_position(stage=s, x=i.x, y=i.y, z=i.z)

    return s


@register(collections=["collection"])
def get_files_ff(
    dataset: DatasetFragment,
) -> List[OmeroFileFragment]:
    """Get all Omerfiles in Dataset

    Gets the files in an dataset at the time of the request
    """
    print(dataset.omerofiles)

    return [file for file in dataset.omerofiles if file is not None]


class DataKind(Enum):
    ASTRONAUT = partial(
        data.astronaut
    )  # partial needed to make it register as an enum value
    BRAIN = partial(data.brain)
    BRICK = partial(data.brick)
    CAMERA = partial(data.camera)
    CAT = partial(data.cat)
    CELL = partial(data.cell)
    CELLS_3D = partial(data.cells3d)
    CHECKERBOARD = partial(data.checkerboard)
    CHELSEA = partial(data.chelsea)
    CLOCK = partial(data.clock)
    COFFEE = partial(data.coffee)
    COINS = partial(data.coins)
    COLORWHEEL = partial(data.colorwheel)
    EAGLE = partial(data.eagle)
    GRASS = partial(data.grass)
    GRAVEL = partial(data.gravel)
    HORSE = partial(data.horse)
    HUBBLE_DEEP_FIELD = partial(data.hubble_deep_field)
    HUMAN_MITOSIS = partial(data.human_mitosis)
    IMMUNOHISTOCHEMISTRY = partial(data.immunohistochemistry)
    KIDNEY = partial(data.kidney)
    LILY = partial(data.lily)
    LOGO = partial(data.logo)
    MICROANEURYSMS = partial(data.microaneurysms)
    MOON = partial(data.moon)
    NICKEL_SOLIDIFICATION = partial(data.nickel_solidification)
    PAGE = partial(data.page)
    PROTEIN_TRANSPORT = partial(data.protein_transport)
    RETINA = partial(data.retina)
    ROCKET = partial(data.rocket)
    SHEPP_LOGAN_PHANTOM = partial(data.shepp_logan_phantom)
    SKIN = partial(data.skin)
    TEXT = partial(data.text)
    VORTEX = partial(data.vortex)

TWOD_DATA = set(
    [
        DataKind.BRICK,
        DataKind.CAMERA,
        DataKind.CELL,
        DataKind.CLOCK,
        DataKind.COINS,
        DataKind.EAGLE,
        DataKind.GRASS,
        DataKind.GRAVEL,
        DataKind.CHECKERBOARD,
        DataKind.HORSE,
        DataKind.HUMAN_MITOSIS,
        DataKind.MICROANEURYSMS,
        DataKind.MOON,
        DataKind.PAGE,
        DataKind.SHEPP_LOGAN_PHANTOM,
        DataKind.TEXT,
        DataKind.VORTEX,

    ]
)
THREED_DATA = set(
    [
        DataKind.BRAIN,
    ]
)
TWOD_D_RGB_DATA = set(
    [
        DataKind.ASTRONAUT,
        DataKind.CAT,
        DataKind.CHELSEA,
        DataKind.COFFEE,
        DataKind.COLORWHEEL,
        DataKind.HUBBLE_DEEP_FIELD,
        DataKind.IMMUNOHISTOCHEMISTRY,
        DataKind.RETINA,
        DataKind.ROCKET,
        DataKind.SKIN,
    ]
)
THREED_CHANNEL_DATA = set([DataKind.CELLS_3D])
ZXYC_DATA = set([DataKind.KIDNEY])
XYC_DATA = set([DataKind.LILY, DataKind.LOGO])
TXY_DATA = set([DataKind.NICKEL_SOLIDIFICATION])
TCXY_DATA = set([DataKind.PROTEIN_TRANSPORT])


@register(collections=["generator"])
def generate_test_image(
    kind: DataKind, attach_meta: bool = True
) -> RepresentationFragment:
    """Generate Test Image"""

    data = kind.value()
    print(data.shape)

    if kind in TWOD_DATA:
        data = xr.DataArray(data, dims=["y", "x"])
        variety = RepresentationVariety.VOXEL
    elif kind in THREED_DATA:
        data = xr.DataArray(data, dims=["z", "y", "x"])
        variety = RepresentationVariety.VOXEL
    elif kind in TWOD_D_RGB_DATA:
        data = xr.DataArray(data, dims=["y", "x", "c"])
        variety = RepresentationVariety.RGB
    elif kind in THREED_CHANNEL_DATA:
        data = xr.DataArray(data, dims=["z", "c", "y", "x"])
        variety = RepresentationVariety.VOXEL
    elif kind in ZXYC_DATA:
        data = xr.DataArray(data, dims=["z", "x", "y", "c"])
        variety = RepresentationVariety.VOXEL
    elif kind in XYC_DATA:
        data = xr.DataArray(data, dims=["x", "y", "c"])
        variety = RepresentationVariety.VOXEL
    elif kind in TXY_DATA:
        data = xr.DataArray(data, dims=["t", "x", "y"])
        variety = RepresentationVariety.VOXEL
    elif kind in TCXY_DATA:
        data = xr.DataArray(data, dims=["t", "c", "x", "y"])
        variety = RepresentationVariety.VOXEL
    else:
        raise Exception("Unknown Data Kind")

    omero = None
    views = []

    if attach_meta:
        physical_size = None
        if kind == DataKind.CELLS_3D:
            nuclei_channel = create_channel(name="Nuclei")
            cell_membrane_channel = create_channel(name="Cell Membrane")

            physical_size = PhysicalSizeInput(x=0.29, y=0.29, z=0.29)

            views.append(
                RepresentationViewInput(
                    cMin=0,
                    cMax=0,
                    channel=cell_membrane_channel,
                )
            )
            views.append(
                RepresentationViewInput(
                    cMin=1,
                    cMax=1,
                    channel=nuclei_channel,
                )
            )

        if kind == DataKind.CELL:
            physical_size = PhysicalSizeInput(x=0.107, y=0.107)

        if kind == DataKind.KIDNEY:
            channel_1 = create_channel(name="Channel 1", emission_wavelength=450)
            channel_2 = create_channel(name="Channel 2", emission_wavelength=515)
            channel_3 = create_channel(name="Channel 3", emission_wavelength=605)

            physical_size = PhysicalSizeInput(x=1.24, y=1.24, z=1.25)

            views.append(
                RepresentationViewInput(
                    cMin=0,
                    cMax=0,
                    channel=channel_1,
                )
            )
            views.append(
                RepresentationViewInput(
                    cMin=1,
                    cMax=1,
                    channel=channel_2,
                )
            )
            views.append(
                RepresentationViewInput(
                    cMin=2,
                    cMax=2,
                    channel=channel_3,
                )
            )

        if kind == DataKind.LILY:
            physical_size = PhysicalSizeInput(x=1.24, y=1.24)



        omero = OmeroRepresentationInput(physicalSize=physical_size)

    return from_xarray(
        data,
        name=f"Test Image {kind.name}",
        tags=[f"test-image-{kind.name}"],
        variety=variety,
        omero=omero,
        views=views,
    )


@register(collections=["segmentation"])
def mark_clusters_of_size_rectangle(
    rep: RepresentationFragment,
    distance: float,
    min_size: int,
    max_size: Optional[int],
    c: Optional[int] = 0,
    t: Optional[int] = 0,
    z: Optional[int] = 0,
    limit: Optional[int] = None,
) -> List[ROIFragment]:
    """Mark Clusters

    Takes a masked image and marks rois based on dense clusters of a certain size 
    and density of their center of mass

    Args:
        rep (RepresentationFragment): The image to label outliers for
        distance (float): The distance between points in a cluster (eps in DBSCAN)
        min_size (int): The minimum size of a cluster (min_samples in DBSCAN)
        max_size (Optional[int]): The maximum size of a cluster (threshold for number of labels in a cluster)
        c (Optional[int], optional): The channel to use. Defaults to 0.
        t (Optional[int], optional): The timepoint to use. Defaults to 0.
        z (Optional[int], optional): The z-slice to use. Defaults to 0.
        limit (Optional[int], optional): The maximum number of clusters to return. Defaults to None.


    Returns:
        List[ROIFragment]: The rois for the clusters
    """
    assert (
        rep.variety == RepresentationVariety.MASK
    ), "Only mask representations are supported"

    x = rep.data.sel(c=c, t=t, z=z).compute().data

    # %%
    centroids = ndimage.center_of_mass(x, x, range(1, x.max() + 1))
    centroids = np.array(centroids)

    dbscan = DBSCAN(eps=distance, min_samples=min_size).fit(centroids)
    dblabels = dbscan.labels_
    n_clusters_ = len(set(dblabels)) - (1 if -1 in dblabels else 0)
    n_noise_ = list(dblabels).count(-1)

    log(f"Estimated number of clusters: {n_clusters_}")
    log(f"Estimated number of noise points:  {n_noise_}")

    if limit is not None:
        if n_clusters_ > limit:
            log(f"Limiting number of clusters to {limit}")
            n_clusters_ = limit
    rois = []

    for cluster in range(0, n_clusters_):
        centrois = centroids[dblabels == cluster]
        if max_size is not None and len(centrois) > max_size:
            continue


        in_labels = []

        for centoid in centrois:
            label_at_centroid = x[int(centoid[0]), int(centoid[1])]
            if label_at_centroid != 0:
                in_labels.append(label_at_centroid)

        if len(in_labels) == 0:
            continue


        mask = np.isin(x, in_labels)
        y_coords, x_coords = np.where(mask)

        y1 = np.min(y_coords)
        x1 = np.min(x_coords)
        y2 = np.max(y_coords)
        x2 = np.max(x_coords)


        roi = create_roi(
            rep,
            vectors=[
                InputVector(x=x1, y=y1, z=z, t=t, c=c),
                InputVector(x=x2, y=y1, z=z, t=t, c=c),
                InputVector(x=x2, y=y2, z=z, t=t, c=c),
                InputVector(x=x1, y=y2, z=z, t=t, c=c),
            ],
            type=RoiTypeInput.RECTANGLE,
            label="size: {}".format(len(centrois)),
        )
        rois.append(roi)

    return rois




@register(collections=["organization"])
def merge_tables(table: List[TableFragment]) -> TableFragment:
    """Merge Tables

    Merges a list of tables into a single table

    Args:
        table (List[TableFragment]): The tables to merge

    Returns:
        TableFragment: The merged table
    """
    rep_origins = []
    for t in table:
        for o in t.rep_origins:
            rep_origins.append(o)

    frames = []

    for t in table:
        same = t.data
        same["origin"] = t.id
        frames.append(same)


    new_dataframe = pd.concat(frames, ignore_index=True)

    return from_df(new_dataframe, name="Merged Table", rep_origins=rep_origins)

