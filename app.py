from functools import reduce
import logging
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
from rich.logging import RichHandler
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
    from_xarray,
    get_representation,
    InputVector,
)
from arkitekt import register, log
from functools import partial


class Colormap(Enum):
    VIRIDIS = partial(cm.viridis)  # partial needed to make it register as an enum value
    PLASMA = partial(cm.plasma)


@register()
def array_to_image(
    rep: RepresentationFragment,
    rescale=True,
    max=True,
    cm: Colormap = Colormap.VIRIDIS,
) -> ThumbnailFragment:
    """Thumbnail Image

    Generates THumbnail for the Image

    Args:
        rep (Representation): The to be converted Image
        rescale (bool, optional): SHould we rescale the image to fit its dynamic range?. Defaults to True.
        max (bool, optional): Automatically z-project if stack. Defaults to True.
        cm: (Colormap, optional): The colormap to use. Defaults to Colormap.VIRIDIS.

    Returns:
        Thumbnail: The Thumbnail
    """

    array = rep.data

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


@register()
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


@register()
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


@register()
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


@register()
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


@register()
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


@register()
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


@register()
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


@register()
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


class ThresholdMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


@register()
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


@register()
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
    m = rep.data.max(dim="z")

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


@register()
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


@register()
def otsu_thresholding(
    rep: RepresentationFragment,
    gaussian_blur: bool = False,
) -> RepresentationFragment:
    """Otsu Thresholding

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
                img = x.sel(c=c, z=z, t=t).data
                if gaussian_blur:
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                print(normed)
                threshold, image = cv2.threshold(
                    normed,
                    0,
                    1,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                print(image, threshold)
                thresholded[c, t, z, :, :] = image

    return from_xarray(
        thresholded,
        name=f"Otsu Thresholded {rep.name}",
        tags=["otsu-thresholded"],
        variety=RepresentationVariety.MASK,
        origins=[rep],
    )


@register()
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

    while smart_image.omero is None or smart_image.omero.position is None:
        smart_image = get_representation(smart_image.origins[0])
        assert (
            smart_image.shape == roi.representation.shape
        ), "Could not find a matching position is not in the same space as the original (probably through a downsampling, or cropping)"

    omero = smart_image.omero
    affine_transformation = omero.affine_transformation
    shape = smart_image.shape
    position = smart_image.omero.position

    # calculate offset between center of roi and center of image
    print(omero.position)
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
        np.array([omero.position.x, omero.position.y, omero.position.z]) + vec
    )

    print(vec)

    print("Affine", affine_transformation)

    return create_position(
        stage=position.stage,
        name=f"Position of {roi.label or 'Unknown ROI'}",
        x=new_pos_x,
        y=new_pos_y,
        z=new_pos_z,
    )


@register()
def create_stage_from_name(
    name: str,
) -> StageFragment:
    """Create New Stage

    Creates a new stage with the given name

    """

    return create_stage(name=name)


@register()
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


@register()
def get_files_ff(
    dataset: DatasetFragment,
) -> List[OmeroFileFragment]:
    """Get all Omerfiles in Dataset

    Gets the files in an dataset at the time of the request
    """
    print(dataset.omerofiles)

    return [file for file in dataset.omerofiles if file is not None]