# general python imports
import numpy as np
from pathlib import Path
from typing import Callable, Sequence, Union, List, Optional, overload

# omniverse imports
from isaac_stage import prims
from omni.isaac.core.utils.prims import define_prim
from isaac_stage import utils


class Asset(object):
    """An instancer for a particular .usd asset.

    Attributes:
        __file_path (str): The file path of the asset.
        __name (str): The base name of the asset. Requires that this name is unique within the scene.
        __count (int): The count of instances of the asset (used to ensure unique names)
        __applier (str -> None | None): A function that is called on the prim path of the asset after it is generated.
            NOTE: Typically useful to apply a physics material, etc.            
        area (float): The area of the asset's bounding box.

    Methods:
        __init__(self, asset_file_path : Path, asset_scale : float, applier : str -> None | None, area_factor : float = None): Initializes the Asset with the given file path, scale, and physics.
        insert(self, parent_prim_path, translation, rotation, rotation_order, scale) -> str: Inserts an instance of the asset into the scene and returns its path.
    """
    def __init__(self, asset_file_path : str, asset_scale : float, area_factor : float=1.0, applier : Optional[Callable[[str], None]]=None):
        """Initializes the Asset with the given file path.

        Note:
            The asset's name becomes the last part of its file path without the suffix, i.e., the stem.
            ~/.../assets/this_is_the_asset_name.usd*
            
        Requires:
            All assets to be added to the scene have unique stems. 
            Because the name is taken from the stem, dummy.usd and dummy.usdz would cause name collisions. 

        Args:
            asset_file_path (Path): The file path of the asset.
            asset_scale (float): The scale factor of the asset (in all directions equally).
            applier (str -> None | None): A function that applies a physics material (or whatever you want) to the prim path.
            area_factor (float)=None: An optional corrective factor if the bounding box overestimates the effective area of the asset.
        """
        self.__file_path : Path = Path(asset_file_path).resolve()
        self.__name = self.__file_path.stem
        self.__count = 0
        self.asset_scale = asset_scale
        self.applier = applier
        self.area_factor = area_factor

        def calculate_area():
            temp_prim = self.insert() # TODO: Current area check requires a prim be created/destroyed, ideally this can be done without modifying the scene.
            bounding_box = utils.get_context().compute_path_world_bounding_box(temp_prim)           
            area = self.area_factor * np.abs(bounding_box[0][0] - bounding_box[1][0]) * np.abs(bounding_box[0][1] - bounding_box[1][1])
            prims.delete(temp_prim)
            print(F"\t└─> {self.__name}, area={area}m^2")
            return area

        self.area = calculate_area()

    def get_name(self):
        return self.__name

    def insert(self, 
               parent_prim_path : Optional[str]=None, 
               translation      : Sequence[float]=(0.0,0.0,0.0), 
               rotation         : Sequence[float]=(0.0,0.0,0.0), 
               rotation_order   : Sequence[float]=(0,1,2), 
               scale            : Sequence[float]=(1,1,1)):
        """Inserts this asset as prim into the current scene and returns its path.

        Args:
            parent_prim_path (str)=None: The path of the parent prim. If None, the prim ("/World/") is used as the parent prim.
            translation (float subscriptable @ 0,1,2)=(0,0,0): The translation values in x, y, and z axes, respectively.
            rotation (float subscriptable @ 0,1,2)=(0,0,0) : The rotation values in x, y, and z axes, respectively.
            rotation_order (int subscriptable @ 0,1,2)=(0,1,2): The order of rotations in x, y, and z axes, respectively.
            scale (float subscriptable @ 0,1,2)=(1,1,1): The scaling values in x, y, and z axes, respectively.

        Requires:
            No one else is spawning this asset or assets with the same name (in the parent_prim_path) without using this interface.
        
        Returns:
            str: The path of the newly created prim.
        """
        parent_prim_path = "/World/" if parent_prim_path is None else parent_prim_path
        asset_prim_path = parent_prim_path + self.__name + F"_{self.__count}"
        self.__count += 1
        asset_prim = define_prim(asset_prim_path,"Xform")
        asset_prim.GetReferences().AddReference(str(self.__file_path))

        prims.transform(asset_prim_path, translation, rotation, rotation_order, scale=(self.asset_scale * scale[0], self.asset_scale * scale[1], self.asset_scale * scale[2])) 
        # Soft TODO: Maybe use an xform wrapping the reference instead.

        if self.applier:
            self.applier(asset_prim_path)

        return asset_prim_path 


class AssetManager(object):
    """A class for managing assets in a given directory.

    Methods:
        __init__(self): Initializes the AssetManager with no registered assets.
        sample_asset
        register_asset(self, asset : Asset)
        register_asset_list(self,  asset_list : List[Asset])
        register_path(self, asset_path : str,  asset_scale : float, area_factor : float=None, applier : Union[Callable[[str], None], None]=None)
        register_path_list(self, asset_path_list : List[str], asset_scale : float, area_factor : float=None, applier : Union[Callable[[str], None], None]=None)
        register_directory(self, asset_directory : str, recurse : bool, asset_scale : float, area_factor : float=None, applier : Union[Callable[[str], None], None]=None)
        register_directory_list(self, asset_directories : List[str], recurse : bool, asset_scale : float, area_factor : float=None, applier : Union[Callable[[str], None], None]=None)

    
    Requires:
        All registered assets have unique names. Per this version, dummy.usdz and dummy.usd cause a collision when registered together. 
    """
    def __init__(self):
        """Initializes an AssetManager."""
        self.registered_assets : np.ndarray = np.array([])

    def sample_asset(self, weight_of_asset : Callable[[Asset],float] = lambda asset : 1 / (1 + np.sqrt(asset.area))) -> Asset :
        """Provides an asset sampled with probability proportionate to the given weights.

        Args:
            weight_of_asset (Callable[[Asset],float]): A function that takes an Asset and returns a weight for it. Default is 1 / (1 + sqrt(asset.area)).

        Returns:
            Asset: A sampled asset with probability proportional to the given weights (handles normalization).
        """
        weights = np.array([weight_of_asset(asset) for asset in self.registered_assets])
        weights = weights / sum(weights)
        return np.random.choice(self.registered_assets, p=weights)

    @overload
    def register(self, asset: Asset) -> None: ...
        
    @overload
    def register(self, asset: List[Asset]) -> None: ...
        
    @overload
    def register(self, asset: str, asset_scale: float, area_factor: Optional[float] = None, applier: Optional[Callable[[str], None]] = None) -> None: ...
        
    @overload
    def register(self, asset: List[str], asset_scale: float, area_factor: Optional[float] = None, applier: Optional[Callable[[str], None]] = None) -> None: ...
        
    @overload
    def register(self, asset_directory: str, recurse: bool, asset_scale: float, area_factor: Optional[float] = None, applier: Optional[Callable[[str], None]] = None) -> None: ...
        
    @overload
    def register(self, asset_directory: List[str], recurse: bool, asset_scale: float, area_factor: Optional[float] = None, applier: Optional[Callable[[str], None]] = None) -> None: ...

    def register(self, asset, recurse=None, asset_scale=None, applier=None, area_factor=None):
        """Registers the asset at the asset_path with an optional physics material.

        Overload Summary:
            | asset input implementations | recurse | asset_scale | applier            | area_factor    |
            |-------------------------------------------------------------------------------------------|
            | asset : Asset               | None    | None        | None               | None           |
            | asset_list : Asset list     | None    | None        | None               | None           |
            | asset_path : str            | None    | None        | str -> None | None | float = 1.0    |
            | asset_path_list : str list  | None    | None        | str -> None | None | float = 1.0    |
            | asset_directory : str       | bool    | None        | str -> None | None | float = 1.0    |
            | asset_dir_list :  str list  | bool    | None        | str -> None | None | float = 1.0    |

        Args:
            input (Asset | Asset list | str | str list)
            recurse (bool): Whether or not to continue down subdirectories to find assets.
            asset_directory (str | str list): The path of a .usd* asset to register.
            asset_scale (float): The scale factor applied to the asset registered (useful for converting between units).
            area_factor (float): The corrective factor to apply to the area..useful if the bounding box is a mis-estimation of the effective asset area.
            applier ((str -> None) optional): A function that applies a physics material (or whatever you want) given the prim path.
        """
        if isinstance(asset, Asset):
            print(f"{asset} asset")
            self.registered_assets = np.append(self.registered_assets, asset)
        elif isinstance(asset, list) and all(isinstance(a, Asset) for a in asset):
            print(f"{asset} (asset list)")
            for asset in asset:
                self.registered_assets = np.append(self.registered_assets, asset)
        elif isinstance(asset, str) and asset_scale is not None:
            asset_path = Path(asset).resolve()
            if asset_path.is_file() and ".usd" in asset_path.suffix:
                print(f"{asset_path} (asset path)")
                self.registered_assets = np.append(self.registered_assets, Asset(str(asset_path), asset_scale, area_factor or 1.0, applier))
            elif asset_path.is_dir() and recurse is not None:
                print(f"{asset_path} (asset directory)")
                for item in asset_path.iterdir():
                    full_path = Path(asset_path, item)
                    if full_path.is_file() and ".usd" in full_path.suffix:
                        print(f"\t│ Scanning Asset: {full_path}")
                        self.registered_assets = np.append(self.registered_assets, Asset(str(full_path), asset_scale, area_factor or 1.0, applier))
                    elif full_path.is_dir() and recurse:
                        self.register(str(full_path), recurse, asset_scale, applier, area_factor)
        elif isinstance(asset, list) and all(isinstance(a, str) for a in asset) and asset_scale is not None:
            print(f"{asset} (asset path list)")
            for asset_path in asset:
                self.register(str(asset_path), recurse, asset_scale, applier, area_factor)