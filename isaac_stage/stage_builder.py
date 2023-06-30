from typing import Sequence
import numpy as np
from isaac_stage.asset_manager import AssetManager, Asset
from isaac_stage.terrain import Terrain2D
from abc import ABC, abstractmethod
from pathlib import Path
from pxr import *
from isaac_stage.omniverse_utils import save_stage, create_light_dome, create_sky_dynamic, DEFAULT_DYNAMIC_SKIES, create_sky_hdri, DEFAULT_HDRI_SKIES


class StageBuilder(ABC):
    """
    A class for creating environments with terrain and assets.
    """
    def __init__(self):
        pass

    @abstractmethod
    def build_stage(self):
        """Implementation-specific method to construct the stage."""
        pass
    
    def save_stage(self, output_file_path : Path, relative_pathing : bool):
        """
        Saves the current stage to a file. TODO: This method is just a call to omniverse_utils, so not sure if it will stay here.

        Args:
            output_file_path (Path): Path to save the stage at..
                NOTE: Can be saved as .usd, .usda, .usdc, .usdz
            relative_pathing (bool): Whether or not to use relative pathing.

        Example: 
            save_stage(Path("../stages/example_stage.usda"),relative_pathing=True)    NOTE: cwd is '/home/*' for this example
            saves to --> "/home/stages/example_stage.usda"  
        """
        save_stage(output_file_path, relative_pathing)


class ConstructionStageBuilder(StageBuilder):
    """
    A class for creating a construction site environment.

    Attributes:
        xdim (int): The x dimension of the environment.
        ydim (int): The y dimension of the environment.
        terrain (Terrain): The terrain object representing the environment's terrain.
        asset_manager (AssetManager): The asset manager used to populate the environment with assets.
    """
    def __init__(self, xdim : int, ydim : int, terrain : Terrain2D, asset_manager : AssetManager):
        """
        Initializes a new instance of the StageBuilder class.

        Args:
            xdim (int): The x dimension of the environment.
            ydim (int): The y dimension of the environment.
            terrain (Terrain): The terrain object representing the environment's terrain.
            asset_manager (AssetManager): The asset manager used to populate the environment with assets.
        """
        super()
        self.xdim = xdim
        self.ydim = ydim
        self.terrain = terrain
        self.asset_manager = asset_manager
    
    def __populate_assets(self, density, world_translation : Sequence[float]):
        """
        Populates the environment with assets.

        Args:
            density (float): The desired density of assets in the environment.
            world_translation (subscriptable @ 0,1,2) : The world space translation of the environment.
        """
        def calculate_resting_height(center_x, center_y, radius, samples):
            offset_x, offset_y = (2*np.random.random(samples)-1)*radius, (2*np.random.random(samples)-1)*radius
            return np.mean([self.terrain.terrain_fn((center_x + offset_x[i]), (center_y + offset_y[i])) for i in range(samples)])

        def weight_by_bounding_box_area(asset : Asset, small_asset_weight=3, medium_asset_weight=3):
            area = asset.area
            if area <= 3:
                return small_asset_weight
            elif area <= 70:
                return medium_asset_weight
            return 1 #large assets
        
        total_asset_area = 0
        while total_asset_area <= self.xdim * self.ydim * density:
            next_asset = self.asset_manager.sample_asset(weight_by_bounding_box_area)
            radius, area = np.sqrt(next_asset.area), next_asset.area
            x, y = (np.random.random()-0.5)*self.xdim, (np.random.random()-0.5)*self.ydim
            total_asset_area += area
            if "spawn_assets" in self.terrain.get_region_tags(x,y): # NOTE: We still increment area even if no spawn occurs.
                z = calculate_resting_height(x,y,radius=radius,samples=16)
                theta = np.random.random()*360
                next_asset.insert(translation=np.array([x,y,z]) + world_translation, rotation=(0,0,theta)) # NOTE: Not verified, just looks like it does the right thing.

    def build_stage(self, global_offset : Sequence[float], spawn_assets:bool, asset_density:float=0.3):
        """
        Creates the terrain and populates it with assets if enabled.

        Args:
            global_offset (float subscriptable @ 0,1,2): The world space translation of the environment.
            spawn_assets (bool): Whether to spawn assets in the environment.
            asset_density (float): The desired density of assets in the environment.

        Effect:
            Adds terrain/assets to the current usd stage.
        """

        #TODO: Figure out whether dynamic/HDRI skies are supported in Isaac Gym
        #print("Picking random skybox..")
        #create_sky_dynamic(DEFAULT_DYNAMIC_SKIES[np.random.choice(list(DEFAULT_DYNAMIC_SKIES.keys()))])  
        # print("Picking random HDRI skybox..")
        # skybox_dict = DEFAULT_HDRI_SKIES[np.random.choice(["Clear","Cloudy","Storm","Evening","Night"])]
        # skybox_path = skybox_dict[np.random.choice(list(skybox_dict.keys()))]
        # create_sky_hdri(skybox_path)

        #NOTE: For now, light dome.
        print("Creating Sky Dome..")
        create_light_dome(intensity=1000)

        print("Meshing Terrain..")
        self.terrain.create_terrain(self.xdim, self.ydim, global_offset)
        if spawn_assets:
            print("Spawning Assets..")
            self.__populate_assets(asset_density, global_offset)



