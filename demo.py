from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni    

from asset_manager import *
from terrain import *
from stage_builder import *

class IsaacSimRunner(object):
    """Runs omniverse."""
    def __init__(self):
        physics_dt = 1 / 100.0
        render_dt = 1 / 30.0
        self._world = omni.isaac.core.World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

    def run(self) -> None:
        """Step simulation based on rendering downtime"""
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
        return

#-----------------------------------------#
#   basic program to execute generation   #
#-----------------------------------------#

def main():
    """Runs the simulation via the IsaacSimRunner."""

    isaac_sim_runner = IsaacSimRunner()
    simulation_app.update()

    # safe to set up here

    #------------#
    #   assets   #
    #------------#

    # create asset manager
    asset_manager = AssetManager()

    # get asset directories
    assets_from_unknown_store = F"{os.getcwd()}/assets/assets_from_unknown_asset_store"
    assets_ov_industry = [ F"{os.getcwd()}/assets/ov-industrial3dpack-01-100.1.2/{category}" for category in ["Piles", "Racks", "Pallets", "Railing", "Shelves", "Containers"]]

    # register assets
    asset_manager.register_assets_from_directory(assets_from_unknown_store, recurse=True, asset_scale=0.4, applier=make_static_collider)
    asset_manager.register_assets_from_many_directories(assets_ov_industry, recurse=True, asset_scale=0.0133, applier=make_static_collider)

    #-------------#
    #   terrain   #
    #-------------#

    # define terrain function
    terrain = WaveletTerrain(terrain_unit=2)

    #-----------------#
    #   environment   #
    #-----------------#
    
    # create environment object and use it to modify the scene
    environment = StageBuilder(xdim=100,ydim=100,terrain=terrain,asset_manager=asset_manager)
    environment.build_stage(spawn_assets=False, global_offset=[0,0,0],asset_density=0.15)

    # everything after this line happens at until the simulation is closed
    isaac_sim_runner.run() # Process continues until closed by user or exception.

    # post simulation events go here

    simulation_app.close()

if __name__ == "__main__":
    main()
