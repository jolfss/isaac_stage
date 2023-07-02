# general python imports
from pathlib import Path

# pxr
import pxr

# omniverse imports
import omni

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

def open_stage(input_file_path : str):
    """
    Loads a .usd stage.

    NOTE: Needs to be called quite early in the pipeline. Before physics_dt, render_dt, and SimulationContext are initialized.
    NOTE: Not sure why--it looks like it could be a labeling collision with the physics context and other things like it.

    Args:
        input_file_path (str): Path of the stage to open.
            NOTE: Can be .usd, .usda, .usdc, (and maybe .usdz ?)

    Example: 
        open_stage("../stages/example_stage.usda")    NOTE: cwd is '/home/*' for this example
            opens --> "/home/stages/example_stage.usda"  

    """
    input_file_path = str(Path(input_file_path).resolve())
    get_context().open_stage(input_file_path)
         
def save_stage(output_file_path : str):
    """
    Saves the current stage to a file.

    Args:
        output_file_path (str): Path relative to current working directory for where to save the stage.
            NOTE: Can be saved as .usd, .usda, .usdc, .usdz

    Example: 
            save_stage("../stages/example_stage.usda")    NOTE: cwd is '/home/*' for this example
            saves to --> "/home/stages/example_stage.usda"  
    """
    output_file_path = str(Path(output_file_path).resolve())
    get_stage().Export(output_file_path)