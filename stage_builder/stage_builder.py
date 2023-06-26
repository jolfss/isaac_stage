from stage_builder.asset_manager import *
from stage_builder.terrain import Terrain2D
from pxr import *

class StageBuilder(object):
    """
    A class for creating environments with terrain and assets.

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
        self.xdim = xdim
        self.ydim = ydim
        self.terrain = terrain
        self.asset_manager = asset_manager
    

    def build_stage(self, global_offset:Union[MutableSequence[float],Sequence[float]], spawn_assets:bool, asset_density:float=0.3):
        """
        Creates the terrain and populates it with assets if enabled.

        Args:
            global_offset (float subscriptable @ 0,1,2): The world space translation of the environment.
            spawn_assets (bool): Whether to spawn assets in the environment.
            asset_density (float): The desired density of assets in the environment.

        Effect:
            Adds terrain/assets to the current usd stage.
        """
        print("Meshing Terrain..")
        self.terrain.create_terrain(self.xdim, self.ydim, global_offset)
        if spawn_assets:
            print("Spawning Assets..")
            self.__populate_assets(asset_density, global_offset)


    def __populate_assets(self, density, world_translation : Union[MutableSequence[float],Sequence[float]]):
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
