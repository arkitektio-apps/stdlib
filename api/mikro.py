from mikro.rath import MikroRath
from rath.scalars import ID
from enum import Enum
from mikro.traits import Representation
from typing import Literal, Tuple, Optional
from mikro.funcs import aexecute, execute
from pydantic import BaseModel, Field


class FileDatasetFragmentRepresentations(Representation, BaseModel):
    """A Representation is 5-dimensional representation of an image

    Mikro stores each image as sa 5-dimensional representation. The dimensions are:
    - t: time
    - c: channel
    - z: z-stack
    - x: x-dimension
    - y: y-dimension

    This ensures a unified api for all images, regardless of their original dimensions. Another main
    determining factor for a representation is its variety:
    A representation can be a raw image representating voxels (VOXEL)
    or a segmentation mask representing instances of a class. (MASK)
    It can also representate a human perception of the image (RGB) or a human perception of the mask (RGBMASK)

    # Meta

    Meta information is stored in the omero field which gives access to the omero-meta data. Refer to the omero documentation for more information.


    #Origins and Derivations

    Images can be filtered, which means that a new representation is created from the other (original) representations. This new representation is then linked to the original representations. This way, we can always trace back to the original representation.
    Both are encapsulaed in the origins and derived fields.

    Representations belong to *one* sample. Every transaction to our image data is still part of the original acuqistion, so also filtered images are refering back to the sample
    Each iamge has also a name, which is used to identify the image. The name is unique within a sample.
    File and Rois that are used to create images are saved in the file origins and roi origins repectively.


    """

    typename: Optional[Literal["Representation"]] = Field(
        alias="__typename", exclude=True
    )
    id: ID
    name: Optional[str]
    "Cleartext name"

    class Config:
        frozen = True


class FileDatasetFragment(BaseModel):
    typename: Optional[Literal["Dataset"]] = Field(alias="__typename", exclude=True)
    name: str
    "The name of the experiment"
    representations: Optional[Tuple[Optional[FileDatasetFragmentRepresentations], ...]]
    "Associated images through Omero"

    class Config:
        frozen = True


class Get_filedatasetQuery(BaseModel):
    dataset: Optional[FileDatasetFragment]
    'Get a single experiment by ID"\n    \n    Returns a single experiment by ID. If the user does not have access\n    to the experiment, an error will be raised.\n    \n    '

    class Arguments(BaseModel):
        id: ID

    class Meta:
        document = "fragment FileDataset on Dataset {\n  name\n  representations {\n    id\n    name\n  }\n}\n\nquery get_filedataset($id: ID!) {\n  dataset(id: $id) {\n    ...FileDataset\n  }\n}"


async def aget_filedataset(
    id: ID, rath: MikroRath = None
) -> Optional[FileDatasetFragment]:
    """get_filedataset


     dataset:
        A dataset is a collection of data files and metadata files.
        It mimics the concept of a folder in a file system and is the top level
        object in the data model.




    Arguments:
        id (ID): id
        rath (mikro.rath.MikroRath, optional): The mikro rath client

    Returns:
        Optional[FileDatasetFragment]"""
    return (await aexecute(Get_filedatasetQuery, {"id": id}, rath=rath)).dataset


def get_filedataset(id: ID, rath: MikroRath = None) -> Optional[FileDatasetFragment]:
    """get_filedataset


     dataset:
        A dataset is a collection of data files and metadata files.
        It mimics the concept of a folder in a file system and is the top level
        object in the data model.




    Arguments:
        id (ID): id
        rath (mikro.rath.MikroRath, optional): The mikro rath client

    Returns:
        Optional[FileDatasetFragment]"""
    return execute(Get_filedatasetQuery, {"id": id}, rath=rath).dataset
