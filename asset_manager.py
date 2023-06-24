# general python imports
import os
import numpy as np
from pathlib import Path
from typing import Callable, MutableSequence, Sequence, Union, List

# omniverse imports
import omniverse_utils
from omni.isaac.core.utils.prims import define_prim


class Asset(object):
    """An instancer for a particular .usd asset.

    Attributes:
        __file_path (Path): The file path of the asset.
        __name (str): The base name of the asset. Requires that this name is unique within the scene.
        __count (int): The count of instances of the asset (used to ensure unique names)
        __applier (str -> None | None): A function that is called on the prim path of the asset after it is generated.
            NOTE: Typically useful to apply a physics material, etc.            
        area (float): The area of the asset's bounding box.

    Methods:
        __init__(self, asset_file_path : Path, asset_scale : float, applier : str -> None | None ): Initializes the Asset with the given file path, scale, and physics.
        insert(self, parent_prim_path, translation, rotation, rotation_order, scale) -> str: Inserts an instance of the asset into the scene and returns its path.
    """
    def __init__(self, asset_file_path : Path, asset_scale : float, applier : Union[Callable[[str], None], None]):
        """Initializes the Asset with the given file path.

        Args:
            asset_file_path (Path): The file path of the asset.
            asset_scale (float): The scale factor of the asset (in all directions equally).
            applier (str -> None | None): A function that applies a physics material (or whatever you want) to the prim path.
        """
        self.__file_path = asset_file_path
        self.__name = asset_file_path.stem
        self.__count = 0
        self.asset_scale = asset_scale
        self.applier = applier

        def calculate_area():
            temp_prim = self.insert() # TODO: Current area check requires a prim be created/destroyed, ideally this can be done without modifying the scene.
            bounding_box = omniverse_utils.get_context().compute_path_world_bounding_box(temp_prim)           
            area = np.abs(bounding_box[0][0] - bounding_box[1][0]) * np.abs(bounding_box[0][1] - bounding_box[1][1])
            omniverse_utils.delete_prim(temp_prim)
            return area

        self.area = calculate_area()

    def insert(self, 
               parent_prim_path : Union[str, None]=None, 
               translation      : Union[MutableSequence[float],Sequence[float]]=(0.0,0.0,0.0), 
               rotation         : Union[MutableSequence[float],Sequence[float]]=(0.0,0.0,0.0), 
               rotation_order   : Union[MutableSequence[int],Sequence[int]]=(0,1,2), 
               scale            : Union[MutableSequence[float],Sequence[float]]=(1,1,1)):
        """Inserts this asset as prim into the current scene and returns its path.

        Args:
            parent_prim_path (str)=None: The path of the parent prim. If None, the root prim ("/") is used as the parent prim.
            translation (float subscriptable @ 0,1,2)=(0,0,0): The translation values in x, y, and z axes, respectively.
            rotation (float subscriptable @ 0,1,2)=(0,0,0) : The rotation values in x, y, and z axes, respectively.
            rotation_order (int subscriptable @ 0,1,2)=(0,1,2): The order of rotations in x, y, and z axes, respectively.
            scale (float subscriptable @ 0,1,2)=(1,1,1): The scaling values in x, y, and z axes, respectively.

        Returns:
            str: The path of the newly created prim.
        """
        parent_prim_path = "/World/" if parent_prim_path is None else parent_prim_path
        asset_prim_path = parent_prim_path + self.__name + F"_{self.__count}"
        self.__count += 1
        asset_prim = define_prim(asset_prim_path,"Xform")
        asset_prim.GetReferences().AddReference(os.path.join(self.__file_path))

        omniverse_utils.transform_prim(asset_prim_path, translation, rotation, rotation_order, scale=(self.asset_scale * scale[0], self.asset_scale * scale[1], self.asset_scale * scale[2])) 
        # Soft TODO: Maybe use an xform wrapping the reference instead.

        if self.applier:
            self.applier(asset_prim_path)

        return asset_prim_path 


class AssetManager(object):
    """A class for managing assets in a given directory.

    Methods:
        __init__(self): Initializes the AssetManager with no registered assets.
        register_asset(self, asset_directory : str, recurse : bool, applier : str -> None | None) -> None: Registers the assets in the given directory.
        register_many_assets(self, asset_directories : List[str], recurse : bool, applier : str -> None | None) -> None: Registers the assets in all the given directories.
        register_assets_from_directory(self, asset_directory : str, recurse : bool, applier : str -> None | None) -> None: Registers the assets in the given directory.
        register_assets_from_many_directories(self, asset_directories : List[str], recurse : bool, applier : str -> None | None) -> None: Registers the assets in all the given directories.
        sample_asset(self, weight_of_asset : Callable[[Asset],float] = lambda asset : 1 / (1 + asset.area)) -> Asset: Provides a sampled asset with probability proportional to the given weights softmaxed.
    """
    def __init__(self):
        """Initializes an AssetManager."""
        self.registered_assets : np.ndarray = np.array([])

    def register_asset(self, asset_path : str,  asset_scale : float, applier : Union[Callable[[str], None], None]) :
        """Registers the asset at the asset_path with an optional physics material.

        Args:
            asset_directory (str): The path of a usd asset to register.
            asset_scale (float): The scale factor applied to the asset registered (useful for converting between units).
            applier (str -> None | None): A function that applies a physics material (or whatever you want) given the prim path.
        """
        if os.path.isfile(asset_path):
            maybe_asset_path = Path(asset_path)
            if maybe_asset_path.suffix == ".usd":
                self.registered_assets = np.append(self.registered_assets, Asset(maybe_asset_path, asset_scale=asset_scale, applier=applier))

    
    def register_many_assets(self, asset_path_list : List[str], asset_scale : float, applier : Union[Callable[[str], None], None]) :
        """Registers the listed assets from the list of paths with an optional physics material..

        Args:
            asset_directory (str, List[str]): An optional alternative directory or list of directories to register assets from.
            recurse (bool): Whether or not to scan subdirectories; None => False
            asset_scale (float): The scale factor applied to the assets registered (useful for converting between units).
            applier (str -> None | None): A function that applies a physics material (or whatever you want) given the prim path.
        """
        for asset_path in asset_path_list:
            self.register_asset(asset_path=asset_path, asset_scale=asset_scale, applier=applier)


    def register_assets_from_directory(self, asset_directory : str, recurse : bool, asset_scale : float, applier : Union[Callable[[str], None], None]):
        """Registers the assets in the given directory with an optional physics applier, and optionally recurses into subdirectories to find other assets.

        Args:
            asset_directory (str): The directory to register assets from, does not descend subdirectories unless recurse is true.
            recurse (bool): Whether or not to scan subdirectories; None => False
            asset_scale (float): The scale factor applied to the assets registered (useful for converting between units).
            applier (str -> None | None): A function that applies a physics material (or whatever you want) given the prim path.
        """
        assets_to_be_registered = []
        for item in os.listdir(asset_directory):
            full_path = Path(os.path.join(asset_directory, item))
            #print(F"Including Assets from {full_path}")
            if os.path.isfile(full_path):
                maybe_asset_path = Path(full_path)
                if maybe_asset_path.suffix == ".usd":
                    assets_to_be_registered.append(Asset(maybe_asset_path, asset_scale=asset_scale, applier=applier))
            elif os.path.isdir(full_path):
                if recurse:
                    self.register_assets_from_directory(str(full_path), recurse=recurse, asset_scale=asset_scale, applier=applier)
        
        self.registered_assets = np.append(self.registered_assets, assets_to_be_registered)

    
    def register_assets_from_many_directories(self, asset_directories : List[str], recurse : bool, asset_scale : float, applier : Union[Callable[[str], None], None]) :
        """Registers the assets in the given directories with an optional physics material, and optionally recurses to find those in the subdirectories of each.

        Args:
            asset_directory (str, List[str]): An optional alternative directory or list of directories to register assets from.
            recurse (bool): Whether or not to scan subdirectories; None => False
            asset_scale (float) = 1.0: The scale factor applied to the assets registered (useful for converting between units).
            applier (str -> None | None): A function that applies a physics material (or whatever you want) given the prim path.
        """
        for asset_directory in asset_directories:
            self.register_assets_from_directory(asset_directory, recurse=recurse, asset_scale=asset_scale, applier=applier)


    def sample_asset(self, weight_of_asset : Callable[[Asset],float] = lambda asset : 1 / (1 + asset.area)) -> Asset :
        """Provides an asset sampled with probability proportionate to the given weights softmaxxed.

        Args:
            weight_of_asset (Callable[[Asset],float]): A function that takes an Asset and returns a weight for it. Default is 1 / (1 + asset.area).
            TODO: Maybe remove the automatic softmax so people don't have to inverse softmax to get what they want.

        Returns:
            Asset: A sampled asset with probability proportional to the given weights softmaxed.
        """
        weights = [ weight_of_asset(asset) for asset in self.registered_assets ]
        softmax = lambda x : np.exp(x) / np.sum(np.exp(x))
        return np.random.choice(self.registered_assets, p=softmax(weights))
