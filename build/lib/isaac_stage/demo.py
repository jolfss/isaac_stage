from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni    


from omni.isaac.core.simulation_context import SimulationContext

from isaac_stage.asset_manager import *
from isaac_stage.terrain import *
from isaac_stage.stage_builder import StageBuilder

from omni.isaac.orbit.markers import PointMarker



"""
Source: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/reference_python_snippets.html

According to the source above a valid raycast call requires the following: 
 1. the stage contains a physics scene, 
 2. all objects have collision meshes enabled 
 3. the play button has been clicked
"""

class IsaacSimRunner(object):
    """Runs omniverse."""
    def __init__(self):
        physics_dt = 1 / 100.0
        render_dt = 1 / 30.0
        self._world = SimulationContext(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt, backend="torch")

        #points used to debug raytracer
        self._num_points= 100
        self._points = (np.random.rand(self._num_points,3)-0.5)*100
        self._points[:,2] = self._points[:,2]*0+10  #sets the z height to 10
        #raytracer and output storage 
        self._physx_query_interface = omni.physx.get_physx_scene_query_interface()
        self._debug_points: List[PointMarker] = list()

        marker = PointMarker(f"/World/debug_height_sample", self._num_points, radius=0.2)
        marker.set_world_poses(positions=self._points)
        self._debug_points.append(marker)

    
    def rayTraceRandomPoints(self)->None:
        """This method raytraces the distance from points to ground.(i.e. all the points are ray traced towards <0,0,-1> )""" 
        for i in range(self._points.shape[0]):
            output = self._physx_query_interface.raycast_closest(self._points[i],(0.0,0.0,-1.0),100,bothSides=True)
            if output['hit']:
                print(F'HIT: {output["rigidBody"]}') # 'rigidBody' is a field containing the prim path of the object hit
            #print(f'{i}: {output}\n')

    def on_physics_step(self,stepsize:float):
        pass 

    def run(self) -> None:
        """Step simulation based on rendering downtime"""

        self._world.reset()


        self._world.add_physics_callback("IsaacRunner_Physics", self.on_physics_step) # Necessary for physics raytracing calls to be run. See method rayTraceRandomPoints
        
        # change to sim running
        while simulation_app.is_running():
            self.rayTraceRandomPoints()
            self._world.step(render=True)
        return

#-----------------------------------------------------------#
#   basic program demonstrating how the scene is produced   #
#-----------------------------------------------------------#

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

    # register assets with a default material that 1) enables collisions 2) makes them visible to physics raytracing. NOTE: The ground material is static, i.e., objects cannot move. 
    # asset_manager.register_assets_from_directory(assets_from_unknown_store, recurse=True, asset_scale=0.4, applier=omniverse_utils.apply_default_ground_physics_material)
    # asset_manager.register_assets_from_many_directories(assets_ov_industry, recurse=True, asset_scale=0.015, applier=omniverse_utils.apply_default_ground_physics_material)
  
    #-------------#
    #   terrain   #
    #-------------#

    # define terrain function
    terrain = WaveletTerrain(terrain_unit=3)

    #-----------------#
    #   environment   #
    #-----------------#
    
    # create environment object and use it to modify the scene
    environment = StageBuilder(xdim=100,ydim=100,terrain=terrain,asset_manager=asset_manager)
    environment.build_stage(global_offset=[0,0,0],spawn_assets=False,asset_density=0.2)

    # everything after this line happens at until the simulation is closed
    isaac_sim_runner.run() # Process continues until closed by user or exception.

    # post simulation events go here

    simulation_app.close()

if __name__ == "__main__":
    main()
