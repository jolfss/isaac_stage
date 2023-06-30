# general omniverse imports
from typing import List, Tuple, Sequence, Union, Callable
import numpy as np

# pxr
import pxr
from pxr import Vt, Gf, Sdf

# omniverse
import omni
import omni.isaac.core

# isaac stage
from isaac_stage import omniverse_utils

#-------------------------#
#   prim handling utils   #
#-------------------------#

def is_defined(path: str) -> bool:
    """
    Check whether a prim with the given path is defined on the current stage.

    Parameters:
    path (str): The path to the prim.
    """
    prim = omniverse_utils.get_stage().GetPrimAtPath(path)
    return prim.IsValid()

# NOTE: May be deprecated/repurposed because transform is more general and *meant* for a single translation.
# This one uses a *multi* prim method on a single prim.
def translate(prim_path : str, offset : Sequence):
        """
        Translates a USD primitive at the given path by the given offset.

        Args:
            prim_path (str): The path to the USD primitive to translate.
            offset (subscriptable @ 0,1,2): The x, y, z (respectively) offsets to apply to the primitive.
                NOTE: This is not relative to the previous position. Probably should be.

        Raises:
            TypeError: If the prim_path argument is not a string or the offset argument is not a tuple of three floats.
        """
        omni.kit.commands.execute('TransformMultiPrimsSRTCpp',
                count=1,
                paths=[prim_path],
                new_translations=[offset[0], offset[1], offset[2]],
                new_rotation_eulers=[0.0, -0.0, 0.0],
                new_rotation_orders=[0, 1, 2],
                new_scales=[1.0, 1.0, 1.0],
                old_translations=[0.0, 0.0, 0.0],
                old_rotation_eulers=[0.0, -0.0, 0.0],
                old_rotation_orders=[0, 1, 2],
                old_scales=[1.0, 1.0, 1.0],
                usd_context_name='',
                time_code=0.0)
        
def delete(prim_path : str, destructive : bool=True):
        """
        Deletes a USD primitive at the given path.

        Args:
            prim_path (str): The prim path to the prim to delete.
            destructive (bool): Whether to destructively remove the primitive from the stage, or simply mark it as inactive.
        """
        omni.kit.commands.execute('DeletePrims',
                paths=[prim_path],
                destructive=destructive)

def transform(prim_path : str, 
                   translation  : Sequence[float] = (0.0, 0.0, 0.0), 
                   rotation     : Sequence[float] = [0.0, 0.0, 0.0], 
                   rotation_order : Sequence[int]   = (0,1,2), 
                   scale        : Sequence[float] = (1.0, 1.0, 1.0)):
        """
        Transforms a USD primitive using the given translation, rotation, scale, and rotation order.

        Args:
            prim_path (str): The path to the primitive to be transformed.
            translation (float subscriptable @ 0,1,2): The translation to be applied to the primitive.
            rotation (float subscriptable @ 0,1,2): The rotation to be applied to the primitive.
            rotation_order (int subscriptable @ 0,1,2): The order in which the rotations are applied to the primitive.
            scale (Union[List[float], Tuple[float, float, float], np.ndarray]): The scale to be appliedto the primitive.
        """
        gf_translation = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
        gf_rotation = Gf.Vec3d(float(rotation[0]), float(rotation[1]), float(rotation[2]))
        gf_rotation_order = Gf.Vec3i(rotation_order[0], rotation_order[1], rotation_order[2])
        gf_scale = Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2]))
        omni.kit.commands.execute('TransformPrimSRT', #NOTE: There is a TransformMultiPrimsSRTCpp which could be used in place of TransformPrimSRT
                path=Sdf.Path(prim_path),
                new_translation=gf_translation,
                new_rotation_euler=gf_rotation,
                new_rotation_order=gf_rotation_order,
                new_scale=gf_scale,
                old_translation=Gf.Vec3d(0.0, 0.0, 0.0),
                old_rotation_euler=Gf.Vec3d(0.0, 0.0, 0.0),
                old_rotation_order=Gf.Vec3i(0, 1, 2),
                old_scale=Gf.Vec3d(1.0, 1.0, 1.0),
        )

def get_pose(prim_path):
    """
    NOTE: This is a likely suspect if anything seems weird.
    returns: [x, y, z, qx, qy, qz, qw] of prim at prim_path
    """
    stage = omniverse_utils.get_stage()
    if not stage:
        return

    # Get position directly from USD
    prim = stage.GetPrimAtPath(prim_path)

    loc = prim.GetAttribute(
        "xformOp:translate"
    )  # VERY IMPORTANT: change to translate to make it translate instead of scale
    rot = prim.GetAttribute("xformOp:orient")
    rot = rot.Get()
    loc = loc.Get()
    str_nums = str(rot)[1:-1]
    str_nums = str_nums.replace(" ", "")
    str_nums = str_nums.split(",")

    rot = []
    for s in str_nums:
        rot.append(float(s))

    #rot = wvn_utils.euler_of_quat(rot)

    pose = [loc[0], loc[1], loc[2], rot[0], rot[1], rot[2], rot[3]]

    return pose

#-------------------------#
#   prim creation utils   #
#-------------------------#

def create_scope(path : str):
    omni.kit.commands.execute('CreatePrimWithDefaultXform',
        prim_type="Scope",
        prim_path=path,
        attributes={})


def create_trimesh(path : str, 
                    faceVertexCounts : List[int], 
                    faceVertexIndices : np.ndarray, 
                    normals : np.ndarray, 
                    points: Vt.Vec3fArray, 
                    primvars_st : List[Tuple[int,int]]) -> str:
        """
        Writes a new USD triange mesh prim to the stage and returns its prim path.

        Args:
            path (str): The path to create the primitive at.
            faceVertexCounts (list of int): The number of vertices for each face in the mesh. (Should really be [3,3,3,3,... number of triangles]).
            faceVertexIndices (np.ndarray of int): The indices of the vertices that make up each face in the mesh.
            normals (np.ndarray): The vertex normals of the mesh.
            points (Vec3fArray): The vertex positions of the mesh.
            primvars_st (list of tuple of int * int): The texture coordinates of the mesh. (NOTE: Not sure if that's what this really is? Default : [(0,0)] * number of triangles).

        Returns:
            str: The prim path of the mesh.
        """
        omni.kit.commands.execute('CreateMeshPrimWithDefaultXform', prim_type='Cube')
        omni.kit.commands.execute("MovePrim", path_from='/Cube', path_to=path)
        cube_prim = omniverse_utils.get_stage().GetPrimAtPath(path)

        cube_prim.GetAttribute('faceVertexCounts').Set(faceVertexCounts)
        cube_prim.GetAttribute('faceVertexIndices').Set(faceVertexIndices)
        cube_prim.GetAttribute('normals').Set(normals)
        cube_prim.GetAttribute('points').Set(points)
        cube_prim.GetAttribute('primvars:st').Set(primvars_st)

        return path

__global_make_triangle_count=0
def create_triangle(vertices : Sequence[Sequence[float]], parent_prim_path : str, applier : Union[Callable[[str],None],None]) -> str:
    """
    Writes a mesh containing a single triangle into the scene.

    Args:
        vertices ((float subscriptable @ 0,1,2) subscriptable @ 0,1,2): Vertices of the triangle.
        i.e. vertices[0:3][0:3] 
        parent_prim_path (str): The primitive that the triangles should be spawned under.
            NOTE: parent_prim_path='/prim_A/prim_B' --> '/prim_A/prim_B/Triangle_*' 
        applier (str -> None | None): An optional function to apply to the prim once it is created.
    
    Returns:
        The prim path of the triangle created. 
    """
    global __global_make_triangle_count
    while is_defined(F"{parent_prim_path}/Triangle_{__global_make_triangle_count}"):
          __global_make_triangle_count += 1

    prim_path = F"{parent_prim_path}/Triangle_{__global_make_triangle_count}"
    mesh = pxr.UsdGeom.Mesh.Define(omniverse_utils.get_stage(), prim_path)

    # Set up vertex data
    mesh.CreatePointsAttr([Gf.Vec3f(float(vertices[0][0]), float(vertices[0][1]), float(vertices[0][2])), 
                           Gf.Vec3f(float(vertices[1][0]), float(vertices[1][1]), float(vertices[1][2])), 
                           Gf.Vec3f(float(vertices[2][0]), float(vertices[2][1]), float(vertices[2][2]))])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    if applier:
          applier(prim_path)

    return prim_path

__global_make_sphere_count=0
def create_sphere(position : Sequence[float], radius : float, parent_prim_path : str, applier : Union[Callable[[str],None],None]) -> str:
    """
    Writes a sphere into the scene.

    Args:
        position (float subscriptable @ 0,1,2): Center of the sphere. 
            i.e. vertices[0:3][0:3] are all floats
        parent_prim_path (str): The primitive that the spheres should be spawned under.
            NOTE: parent_prim_path='/prim_A/prim_B' --> '/prim_A/prim_B/Sphere_*' 
        applier (str -> None | None): An optional function to apply to the prim once it is created.
    
    Returns:
        The prim path of the sphere created. 
    """
    global __global_make_sphere_count
    while is_defined(F"{parent_prim_path}/Sphere_{__global_make_sphere_count}"):
          __global_make_sphere_count += 1

    prim_path = F"{parent_prim_path}/Sphere_{__global_make_sphere_count}"

    # Define a Sphere
    sphere = pxr.UsdGeom.Sphere.Define(omniverse_utils.get_stage(), prim_path)

    # Set the radius
    sphere.CreateRadiusAttr(radius)

    transform(prim_path, translation=position)

    if applier:
          applier(pxr.Sdf.Path(prim_path))
    return prim_path

def create_light_dome(intensity : float) -> str:
    """Creates a dome light with a given intensity"""
    omni.kit.commands.execute('CreatePrim',
        prim_type='DomeLight',
        attributes={'intensity': intensity, 'texture:format': 'latlong'})

def create_dynamic_sky(dynamic_sky_path : str):
    """Creates a dynamic sky from one of the Omniverse defaults.

    Args:
        dynamic_sky_path (str): Path to the location of a .usd file which satisfies the dynamic sky properties..
        NOTE: The default paths can be easily accessed via the dictionary DEFAULT_DYNAMIC_SKIES[sky_name]
        Possible values of sky_name can be found below.
    
    Effects:
        Creates (or sets--not tested) the sky at /Environment/sky

    Options: Cirrus, ClearSky, CloudySky, CumulusHeavy, CumulusLight, NightSky, Overcast"""
    omni.kit.commands.execute('CreateDynamicSkyCommand',
        sky_url=dynamic_sky_path,
        sky_path='/Environment/sky')
    
def create_hdri_sky(hdri_sky_path : str):
    """Creates a static HDR-imaged sky from one of the Omniverse defaults.

    Args:
        hdri_sky_path (str): Path to the location of a .hdr file for a skybox.
        NOTE: The default paths can be easily accessed via the dictionary DEFAULT_HDRI_SKIES[sky_folder][sky_name]
        Possible values for sky_folder and sky_name can be found below.

    Effects:
        Creates (or sets--not tested) a sky at /Environment/sky

    Options:
        [sky_folder]
            [sky_name]
        Clear
            evening_road_01_4k
            kloppenheim_02_4k
            mealie_road_4k
            noon_grass_4k
            qwantani_4k
            signal_hill_sunrise_4k
            sunflowers_4k
            syferfontein_18d_clear_4k
            venice_sunset_4k
            white_cliff_top_4k
        Cloudy
            abandoned_parking_4k
            champagne_castle_1_4k
            evening_road_01_4k
            kloofendal_48d_partly_cloudy_4k
            lakeside_4k
            sunflowers_4k
            table_mountain_1_4k
        Evening
            evening_road_01_4k
        Indoor
            adams_place_bridge_4k
            autoshop_01_4k
            bathroom_4k
            carpentry_shop_01_4k
            en_suite_4k
            entrance_hall_4k
            hospital_room_4k
            hotel_room_4k
            lebombo_4k
            old_bus_depot_4k
            small_empty_house_4k
            studio_small_04_4k
            surgery_4k
            vulture_hide_4k
            wooden_lounge_4k
            ZetoCG_com_WarehouseInterior2b
            ZetoCGcom_Exhibition_Hall_Interior1
        Night
            kloppenheim_02_4k
            moonlit_golf_4k
        Storm
            approaching_storm_4k
        Studio
            photo_studio_01_4k
            studio_small_05_4k
            studio_small_07_4k"""
    omni.kit.commands.execute('CreateHdriSkyCommand',
        sky_url=hdri_sky_path,
        sky_path='/Environment/sky')