# general python imports
from typing import List, Callable, Tuple

# pxr
import pxr
from pxr import Gf

# omniverse imports
import omni
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim

# isaac stage imports
from isaac_stage import utils

# NOTE: The type of all appliers should be Callable[[str],None]   

#--------------------------------#
#   prim functional operations   #
#--------------------------------#

def apply_appliers(applier_list : List[Callable[[str],None]]) -> Callable[[str],None]:
    """
    Defines an applier over a prim_path (str) given a list of appliers of prim_paths.
    """
    return lambda prim_path : [ applier(prim_path) for applier in applier_list][0]
            
def apply_color_to_prim(color: tuple) -> Callable[[str],None]:
    def __apply_color_to_prim_helper(prim_path : str) -> Tuple[float,float,float] :
        """
        Apply an RGB color to a prim. 

        Parameters:
        prim_path (str): The path to the prim.
        color (tuple): The RGB color to apply as a tuple of three floats.
        """
        stage = utils.get_stage()

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
    return __apply_color_to_prim_helper


def apply_default_ground_physics_material(prim_path : str) -> Callable[[str],None]:
    default_ground_material = PhysicsMaterial(
           "/Materials/groundMaterial", # NOTE: Pick an appropriate place to create this material.
           static_friction=1.0,dynamic_friction=1.0,restitution=0.0)
    
    GeometryPrim(prim_path, collision=True).apply_physics_material(default_ground_material)

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
        usd_prim=utils.get_stage().GetPrimAtPath(prim_path),
        component='PhysicsCollisionAPI')

    # Works #2
    omni.kit.commands.execute('SetStaticCollider',
        path=pxr.Sdf.Path(prim_path),
        approximationShape='none') 
