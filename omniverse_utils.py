# general python imports
import numpy as np
from typing import List, Tuple, Union, Sequence, MutableSequence

# pxr
import pxr
from pxr import Gf, Sdf, Vt

# omniverse imports
import omni
import omni.kit.commands

def context() -> omni.usd.UsdContext:
        """
        Returns the current USD context.

        Returns:
            omni.usd.context.UsdContext: The current USD context.
        """
        return omni.usd.get_context()

def stage() -> pxr.Usd.Stage:
        """
        Returns the current USD stage.

        Returns:
            pxr.Usd.Stage: The current USD stage.
        """
        return omni.usd.get_context().get_stage()

# NOTE: May be deprecated/repurposed because transform is more general and *meant* for a single translation.
# This one uses a *multi* prim method on a single prim.
def translate_prim(prim_path : str, offset : Union[MutableSequence,Sequence]):
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
                   translation  : Union[MutableSequence[float],Sequence[float]] = (0.0, 0.0, 0.0), 
                   rotation     : Union[MutableSequence[float],Sequence[float]] = [0.0, 0.0, 0.0], 
                   rotation_order : Union[MutableSequence[int],Sequence[int]]   = (0,1,2), 
                   scale        : Union[MutableSequence[float],Sequence[float]] = (1.0, 1.0, 1.0)):
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

def trimesh_to_prim(path : str, 
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
        cube_prim = stage().GetPrimAtPath(path)

        cube_prim.GetAttribute('faceVertexCounts').Set(faceVertexCounts)
        cube_prim.GetAttribute('faceVertexIndices').Set(faceVertexIndices)
        cube_prim.GetAttribute('normals').Set(normals)
        cube_prim.GetAttribute('points').Set(points)
        cube_prim.GetAttribute('primvars:st').Set(primvars_st)

        return path

