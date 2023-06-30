# general python imports
import numpy as np
from typing import List, Tuple, Union, Sequence, MutableSequence, Callable
from pathlib import Path

# pxr
import pxr
from pxr import Gf, Sdf, Vt

# omniverse imports
import omni
import omni.kit.commands
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim

#--------------------------#
#   usd management utils   #
#--------------------------#

def get_context() -> omni.usd.UsdContext:
        """
        Returns the current USD context.

        Returns:
            omni.usd.UsdContext: The current USD context.
        """
        return omni.usd.get_context()

def get_stage() -> pxr.Usd.Stage:
        """
        Returns the current USD stage.

        Returns:
            Usd.Stage: The current USD stage.
        """
        return omni.usd.get_context().get_stage()

def open_stage(input_file_path : Path, relative_pathing : bool):
    """
    Loads a .usd stage.

    NOTE: Needs to be called quite early in the pipeline. Before physics_dt, render_dt, and SimulationContext are initialized.
    NOTE: Not sure why--it looks like it could be a labeling collision with the physics context and other things like it.

    Args:
        input_file_path (Path): Path of the stage to open.
            NOTE: Can be .usd, .usda, .usdc, (and maybe .usdz ?)
        relative_pathing (bool): Whether or not to use a relative path.

    Example: 
        open_stage(Path("../stages/example_stage.usda"),relative_pathing=True)    NOTE: cwd is '/home/*' for this example
            opens --> "/home/stages/example_stage.usda"  

    """
    if relative_pathing:
        get_context().open_stage(str(Path(Path.cwd(),input_file_path).resolve()))
    else:
        get_context().open_stage(str(input_file_path.resolve()))
         
def save_stage(output_file_path : Path, relative_pathing : bool):
    """
    Saves the current stage to a file.

    Args:
        output_file_path (Path): Path relative to current working directory for where to save the stage.
            NOTE: Can be saved as .usd, .usda, .usdc, .usdz
        relative_pathing (bool): Whether or not to use a relative path.

    Example: 
        save_stage(Path("../stages/example_stage.usda"),relative_pathing=True)    NOTE: cwd is '/home/*' for this example
            saves to --> "/home/stages/example_stage.usda"  
    """
    if relative_pathing:
        get_stage().Export(str(Path(Path.cwd(),output_file_path).resolve()))
    else:
        get_stage().Export(str(output_file_path.resolve()))

def is_prim_defined(path: str) -> bool:
    """
    Check whether a prim with the given path is defined on the current stage.

    Parameters:
    path (str): The path to the prim.
    """
    prim = get_stage().GetPrimAtPath(path)
    return prim.IsValid()

# NOTE: May be deprecated/repurposed because transform is more general and *meant* for a single translation.
# This one uses a *multi* prim method on a single prim.
def translate_prim(prim_path : str, offset : Sequence):
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
        
def delete_prim(prim_path : str, destructive : bool=True):
        """
        Deletes a USD primitive at the given path.

        Args:
            prim_path (str): The prim path to the prim to delete.
            destructive (bool): Whether to destructively remove the primitive from the stage, or simply mark it as inactive.
        """
        omni.kit.commands.execute('DeletePrims',
                paths=[prim_path],
                destructive=destructive)

def transform_prim(prim_path : str, 
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
    stage = get_stage()
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

def create_prim_trimesh(path : str, 
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
        cube_prim = get_stage().GetPrimAtPath(path)

        cube_prim.GetAttribute('faceVertexCounts').Set(faceVertexCounts)
        cube_prim.GetAttribute('faceVertexIndices').Set(faceVertexIndices)
        cube_prim.GetAttribute('normals').Set(normals)
        cube_prim.GetAttribute('points').Set(points)
        cube_prim.GetAttribute('primvars:st').Set(primvars_st)

        return path

__global_make_triangle_count=0
def create_prim_triangle(vertices : Union[
      MutableSequence[MutableSequence[float]],
      MutableSequence[Sequence[float]],
      Sequence[MutableSequence[float]],
      Sequence[Sequence[float]]],
      applier : Union[Callable[[str],None],None]) -> str:
    """
    Writes a mesh containing a single triangle into the scene.

    Args:
        vertices ((float subscriptable @ 0,1,2) subscriptable @ 0,1,2): Vertices of the triangle.
        i.e. vertices[0:3][0:3] 
        applier (str -> None | None): An optional function to apply to the prim once it is created.
    
    Returns:
        The prim path of the triangle created. 
    """
    global __global_make_triangle_count
    while is_prim_defined(F"/World/Triangle_{__global_make_triangle_count}"):
          __global_make_triangle_count += 1

    prim_path = F"/World/Triangle_{__global_make_triangle_count}"
    mesh = pxr.UsdGeom.Mesh.Define(get_stage(), prim_path)

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
def create_prim_sphere(position : Sequence[float], radius : float, applier : Union[Callable[[str],None],None]) -> str:
    """
    Writes a sphere into the scene.

    Args:
        position (float subscriptable @ 0,1,2): Center of the sphere.
        i.e. vertices[0:3][0:3] 
        applier (str -> None | None): An optional function to apply to the prim once it is created.
    
    Returns:
        The prim path of the sphere created. 
    """
    global __global_make_sphere_count
    while is_prim_defined(F"/World/Sphere_{__global_make_sphere_count}"):
          __global_make_sphere_count += 1

    prim_path = F"/World/Sphere_{__global_make_sphere_count}"

    # Define a Sphere
    sphere = pxr.UsdGeom.Sphere.Define(get_stage(), prim_path)

    # Set the radius
    sphere.CreateRadiusAttr(radius)

    # Create a transform
    xform = pxr.UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]),float(position[1]),float(position[2])))

    if applier:
          applier(pxr.Sdf.Path(prim_path))
    return prim_path

def create_light_dome(intensity : float) -> str:
    """Creates a dome light with a given intensity"""
    omni.kit.commands.execute('CreatePrim',
        prim_type='DomeLight',
        attributes={'intensity': intensity, 'texture:format': 'latlong'})

# Omniverse's default dynamic skyboxes.
DEFAULT_DYNAMIC_SKIES= {sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Dynamic/{sky_name}.usd" \
                                   for sky_name in ["Cirrus", "ClearSky", "CloudySky", "CumulusHeavy", "CumulusLight", "NightSky", "Overcast"]}
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
    
# Omniverse's default HDR-imaged, static skyboxes.
__DEFAULT_HDRI_CLEAR_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Clear/{sky_name}.hdr" \
                                   for sky_name in ["evening_road_01_4k", "kloppenheim_02_4k", "mealie_road_4k", "noon_grass_4k", 
                                                    "qwantani_4k", "signal_hill_sunrise_4k", "sunflowers_4k","syferfontein_18d_clear_4k",
                                                    "venice_sunset_4k","white_cliff_top_4k"]}
__DEFAULT_HDRI_CLOUDY_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Cloudy/{sky_name}.hdr" \
                                   for sky_name in ["abandoned_parking_4k", "champagne_castle_1_4k", "evening_road_01_4k", 
                                                    "kloofendal_48d_partly_cloudy_4k", "lakeside_4k", "sunflowers_4k", 
                                                    "table_mountain_1_4k"]}
__DEFAULT_HDRI_EVENING_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Evening/{sky_name}.hdr" \
                                   for sky_name in ["evening_road_01_4k"]}
__DEFAULT_HDRI_INDOOR_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Indoor/{sky_name}.hdr" \
                                   for sky_name in ["adams_place_bridge_4k","autoshop_01_4k","bathroom_4k","carpentry_shop_01_4k",
                                                    "en_suite_4k","entrance_hall_4k","hospital_room_4k","hotel_room_4k","lebombo_4k",
                                                    "old_bus_depot_4k","small_empty_house_4k","studio_small_04_4k","surgery_4k",
                                                    "vulture_hide_4k","wooden_lounge_4k","ZetoCG_com_WarehouseInterior2b",
                                                    "ZetoCGcom_Exhibition_Hall_Interior1"]}
__DEFAULT_HDRI_NIGHT_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Night/{sky_name}.hdr" \
                                   for sky_name in ["kloppenheim_02_4k","moonlit_golf_4k"]}
__DEFAULT_HDRI_STORM_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Storm/{sky_name}.hdr" \
                                   for sky_name in ["approaching_storm_4k"]}
__DEFAULT_HDRI_STUDIO_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Studio/{sky_name}.hdr" \
                                   for sky_name in ["photo_studio_01_4k","studio_small_05_4k","studio_small_07_4k"]}
# Primary dictionary containing the above sub-dictionaries.
DEFAULT_HDRI_SKIES = { "Clear" : __DEFAULT_HDRI_CLEAR_SKIES,"Cloudy" : __DEFAULT_HDRI_CLOUDY_SKIES,"Evening" : __DEFAULT_HDRI_EVENING_SKIES,
                        "Indoor" : __DEFAULT_HDRI_INDOOR_SKIES,"Night" : __DEFAULT_HDRI_NIGHT_SKIES, "Storm" : __DEFAULT_HDRI_STORM_SKIES,
                        "Studio" : __DEFAULT_HDRI_STUDIO_SKIES,}
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

#--------------------------------#
#   prim functional operations   #
#--------------------------------#

def apply_appliers(applier_list : List[Callable[[str],None]]) -> Callable[[str],None]:
    """
    Defines an applier over a prim_path (str) given a list of appliers of prim_paths.
    """
    return lambda prim_path : [ applier(prim_path) for applier in applier_list][0]
            
def apply_color_to_prim(prim_path: str, color: tuple):
    """
    Apply an RGB color to a prim.

    Parameters:
    prim_path (str): The path to the prim.
    color (tuple): The RGB color to apply as a tuple of three floats.
    """
    stage = get_stage()

    # Access the prim
    prim = stage.GetPrimAtPath(prim_path)

    # Check if the prim exists
    if not prim:
        print(f'Error: No prim at {prim_path}')
        return

    # Get the UsdGeom interface for the prim
    prim_geom = pxr.UsdGeom.Gprim(prim)

    # Create a color attribute if it doesn't exist
    if not prim_geom.GetDisplayColorAttr().HasAuthoredValueOpinion():
        prim_geom.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # Set the color
    else:
        prim_geom.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # # Get the prim
    # stage = get_stage()
    # prim = stage.GetPrimAtPath(prim_path)

    # if not prim or not prim.IsA(pxr.UsdGeom.Mesh):
    #     print(f"No Mesh prim at path: {prim_path}")
    #     return

    # # Create a shader (simple UsdPreviewSurface with constant color)
    # shader = pxr.UsdShade.Shader.Define(stage, f'{prim_path}/ColorShader')
    # shader.CreateIdAttr('UsdPreviewSurface')
    # shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))

    # # Create a Material and connect the shader to its surface terminal
    # material = pxr.UsdShade.Material.Define(stage, f'{prim_path}/ColorMaterial')
    # material.CreateSurfaceOutput().ConnectToSource(shader, 'surface')

    # # Bind the Material to the Mesh
    # pxr.UsdShade.MaterialBindingAPI(prim).Bind(material) 

def apply_default_ground_physics_material(prim_path : str):
    DEFAULT_GROUND_MATERIAL = PhysicsMaterial(
           "/Materials/groundMaterial", # NOTE: Pick an appropriate place to create this material.
           static_friction=1.0,dynamic_friction=1.0,restitution=0.0)
    
    GeometryPrim(prim_path, collision=True).apply_physics_material(DEFAULT_GROUND_MATERIAL)

def __apply_default_static_collider(prim_path:str):
    """
    WARNING:
        This method was deprecated because it is not leveraging the Isaac-Sim interface,
        which is meant to abstract such low-level processes. LEFT FOR CONTEXT. 
    
    Not entirely sure how this works--but it makes an object a static collider.

    Args:
        path (str): The prim path of the prim to add physics properties to.
    """
    
    # Works but Convex Hull (bounding box) for Terrain.
    #utils.setCollider(stage().GetPrimAtPath(prim_path), approximationShape="None")

    # Works #1
    omni.kit.commands.execute('AddPhysicsComponent',
        usd_prim=get_stage().GetPrimAtPath(prim_path),
        component='PhysicsCollisionAPI')

    # Works #2
    omni.kit.commands.execute('SetStaticCollider',
        path=pxr.Sdf.Path(prim_path),
        approximationShape='none') 
