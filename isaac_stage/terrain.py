import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Sequence, Set, Callable

from pxr import Gf, Vt
from isaac_stage import appliers, prims

from isaac_stage import utils


class Terrain(ABC):
    """Abstract base class for generating terrain meshes.

    Attributes:
        terrain_unit (float): The distance between samples. I.e., the 'dx' or 'dy' from one row/column of vertices to the next. 

    Methods:
        terrain_fn(x, y) -> float:
            Return the height of the terrain at (x, y), optionally sensitive to the randomize method.

        randomize(seed) -> None:
            Set or randomize the seed, then modify internal state to produce a new/random terrain. Optionally implemented, may have no effect on generation.

        get_region_tags(x,y) -> Set of str:
            Is a set of implementation-specific tags associated with the (x,y) coordinate of the terrain.

        create_terrain(xdim, ydim, world_translation) -> str (prim path):
            Create a triangle mesh of the terrain with dimensions xdim by ydim, centered at world_translation.
    """

    def __init__(self, terrain_unit : float, applier : Union[Callable[[str],None],None]):
        """Initialize a Terrain with a set generation resolution and optional applier."""
        self.terrain_unit = terrain_unit 
        self.applier = applier

    @abstractmethod
    def randomize(self, seed) -> None :
        """Set or randomize the random seed and do whatever is necessary (modify internal state) to randomize next terrain_fn call."""
        pass

    @abstractmethod
    def terrain_fn(self, x : float, y : float) -> float :
        """The height of the terrain at (x,y), optionally sensitive to the randomize method for parametrized environments."""
        pass
    
    @abstractmethod
    def get_region_tags(self, x, y) -> Set[str]:
        """
        A set of implementation-specific tags associated with the position (x,y) on the terrain.

        Params:
            x (float) : Location on terrain to get tag of.
            y (float)

        Usage:
            For example, the following could be used by an external module for placing assets and spawning.
            terrain_object.get_region_tags(0,0) = {"spawn_location", "no_assets"}
            terrain_object.get_region_tags(0,10)= set()
        """
        return set()
    
    def create_terrain(self, xdim : int, ydim : int, world_translation : Sequence[float]) -> str:
        """
        Creates a triangle mesh of the terrain with dimensions xdim by ydim, centered at world_translation.

        Args:
            xdim (float): The x dimension of the terrain.
            ydim (float): The y dimension of the terrain.
            world_translation (list of float | np.ndarray | float * float * float): The world space translation of the terrain.

        Returns:
            str: The prim path to the generated triangle mesh UsdGeom.
        """
        def create_trimesh() -> Tuple[np.ndarray, np.ndarray]:
            """
            Uses triangles to mesh the terrain function, sampled every terrain_unit.

            Returns:
                Tuple[numpy.ndarray, numpy.ndarray]: A tuple of the vertex positions and triangle indices.
            """
            x_axis = np.linspace(-xdim/2,xdim/2,int(np.ceil(xdim/self.terrain_unit)))
            y_axis = np.linspace(-ydim/2,ydim/2,int(np.ceil(ydim/self.terrain_unit)))
            terrain = np.zeros((len(x_axis),len(y_axis)))
            num_rows = terrain.shape[0]
            num_cols = terrain.shape[1]

            for x in range(num_rows):
                print(F"Meshing Terrain {np.floor(100* x * len(y_axis)/(len(x_axis) * len(y_axis)))}%")
                for y in range(num_cols):
                    terrain[x,y] = self.terrain_fn(x_axis[x],y_axis[y])

            yy, xx = np.meshgrid(y_axis, x_axis)

            vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
            vertices[:, 0] = xx.flatten()
            vertices[:, 1] = yy.flatten()
            vertices[:, 2] = terrain.flatten()
            triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
            for i in range(num_rows - 1):
                ind0 = np.arange(0, num_cols-1) + i*num_cols
                ind1 = ind0 + 1
                ind2 = ind0 + num_cols
                ind3 = ind2 + 1
                start = 2*i*(num_cols-1)
                stop = start + 2*(num_cols-1)
                triangles[start:stop:2, 0] = ind0
                triangles[start:stop:2, 1] = ind3
                triangles[start:stop:2, 2] = ind1
                triangles[start+1:stop:2, 0] = ind0
                triangles[start+1:stop:2, 1] = ind2
                triangles[start+1:stop:2, 2] = ind3

            return vertices, triangles

        def compute_normals(vertices, triangles) -> np.ndarray:
            """
            Computes the normal vectors for the given vertices and triangles.
            Note: When the triangle is labeled counterclockwise as viewed from above, the normal is up.

            Args:
                vertices (numpy.ndarray): The vertex positions as a (N, 3) array.
                triangles (numpy.ndarray): The triangle indices as a (M, 3) array.

            Returns:
                numpy.ndarray: The normal vectors as a (M, 3) array.
            """
            normals = np.zeros((triangles.size, 3))
            for i in range(len(triangles)):
                t = triangles[i]
                v1, v2, v3 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
                normal = np.cross(v2 - v1, v3 - v1)
                normals[3*i:3*i+2, :] = normal
            return normals

        vertices, triangles = create_trimesh()
        faceVertexCounts = [3] * len(triangles)
        faceVertexIndices= triangles.flatten()
        normals = compute_normals(vertices, triangles)
        points = Vt.Vec3fArray([Gf.Vec3f(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices])
        primvars_st = [(0,0)] * triangles.size
        
        terrain_id = 0 # Get unique name for the terrain.
        while utils.get_stage().GetPrimAtPath(F"/terrain_mesh_{terrain_id}"):
            terrain_id += 1

        prim_path =  F"/terrain_mesh_{terrain_id}"
        
        terrain_prim = prims.create_trimesh(prim_path, faceVertexCounts, faceVertexIndices, normals, points, primvars_st)
        prims.translate(prim_path, world_translation)

        # Apply Applier
        if self.applier:
            self.applier(prim_path)

        return terrain_prim
    
    def __failed_create_terrain(self):
        # TODO: If dynamic terrain is needed at some point, using @torch.jit.script to calculate the heights might be worthwhile.
        # NOTE: For now, a terrain_fn of numpy/torch ufuncs should work.
        def create_terrain(self, xdim : int, ydim : int, world_translation : Sequence[float]) -> str:
            
            num_rows, num_cols = int(ydim/self.terrain_unit), int(xdim/self.terrain_unit)
            rows, cols = torch.meshgrid(torch.arange(ydim/self.terrain_unit),torch.arange(xdim/self.terrain_unit), indexing='ij')
            Y = ydim * (rows/(num_rows - 1) - 0.5)
            X = xdim * (cols/(num_cols - 1) - 0.5)

            #------------------------------------------------------------------------------------#
            # NOTE: This is the dream implementation but a batched terrain_fn is hard to make.   #
            # terrain_vec_fn : Callable[[Sequence[float]],float] = lambda position : self.terrain_fn(position[:,:,0],position[:,:,1]) #NOTE: Check if x and y should be flipped because of row column
            XY = torch.stack((Y,X), -1) # TODO: Should have the property that grid[col][row] = [x,y] in world space.
            # Z = terrain_vec_fn(XY)                                                             
            #------------------------------------------------------------------------------------#
            
            Z = torch.zeros((num_rows,num_cols))

            for r in range(num_rows):
                print(F"Sampling Terrain .. {np.floor(100*(r/num_rows))}%")
                for c in range(num_cols):
                    Z[r][c] = self.terrain_fn(XY[r,c][0],XY[r,c][1])
                    
            index = lambda r,c : num_cols * r + c

            vertices = torch.stack((X,Y,Z),-1)
            
            triangles = torch.zeros(size=(2*(num_cols-1)*(num_rows-1),3))                   #   NOTE: traveling counter-clockwise -> out of page (*)
            normals = torch.zeros(size=(2*(num_cols-1)*(num_rows-1),3))                   #   A <--- B
            for r in range(num_rows - 1):                                                   #   | * /  ^
                print(F"Meshing Terrain .. {np.floor(100*(r/num_rows))}%")               #   v  / * |
                for c in range(num_cols -1):                                                #   C ---> D
                    i = 2 * index(r,c)                                                      
                    A, B, C, D =index(r,c), index(r,c+1), index(r+1,c), index(r+1,c+1)
                    vA,vB,vC,vD=vertices[[A,B,C,D]]
                    triangles[ i ,0] = A
                    triangles[ i ,1] = B
                    triangles[ i ,2] = C
                    normals[   i ,:] = torch.cross(vA-vB, vC-vB)
                    triangles[i+1,0] = D
                    triangles[i+1,1] = C
                    triangles[i+1,2] = B
                    normals[  i+1,:] = torch.cross(vD-vC, vB-vC)

            terrain_id = 0 # Get unique name for the terrain.
            while utils.get_stage().GetPrimAtPath(F"/terrain_mesh_{terrain_id}"):
                terrain_id += 1

            prim_path =  F"/terrain_mesh_{terrain_id}"
            
            faceVertexCounts = [3] * len(triangles)
            faceVertexIndices= triangles.flatten()
            primvars_st = [(0,0)] * triangles.size
            terrain_prim = prims.create_trimesh(prim_path, faceVertexCounts, faceVertexIndices, normals, vertices, primvars_st)
            prims.translate(prim_path, world_translation)

            # Apply Applier
            if self.applier:
                self.applier(prim_path)

            return terrain_prim


class WaveletTerrain(Terrain):
    """A terrain made of wavelets with localized damping and a flat patch.

    Methods:
        __init__(self, terrain_unit=0.5, applier=omniverse_utils.apply_default_ground_physics_material, seed=None, xdim=100, ydim=100, amp=8, low=0, high=2, num_rough=80, num_smooth=40, roughness=100, smoothness=70, protect=(0.0, 0.0), protect_radius=10, protect_decay=2):
            Initializes a new instance of the WaveletTerrain class with the specified parameters.

        randomize(self, seed=None):
            Randomizes the wavelet frequencies and placement of roughing and smoothing spots/nodes.

        terrain_fn(self, x, y) -> float:
            Computes the height of the terrain at the given (x, y) coordinates."""

    def __init__(self, 
                 terrain_unit : float = 0.5,
                 applier = appliers.apply_default_ground_physics_material, 
                 seed : Union[int, None] = None,
                 xdim : int = 100, 
                 ydim : int = 100, 
                 amp : float = 8, 
                 low : float = 0, 
                 high: float = 2,
                 num_rough : int = 80, 
                 num_smooth : int = 40, 
                 roughness : int = 100, 
                 smoothness : int = 70,
                 protect : Union[Tuple[float, float], np.ndarray, List[float], None] = (0.0,0.0), 
                 protect_radius : float = 10, 
                 protect_decay : float = 2):
        """
        Parameters:
            terrain_unit (float): The physical gap between vertices in the terrain mesh (dx per row and column)
            applier (str -> None | None): A function to apply upon creation of the terrain prim.
            seed (int): Seed for randomization
                (NOTE: None -> a random seed is chosen)
            xdim,ydim (int): Intended domain for sampling x \\in [-xdim/2,ydim/2], y \\in [-ydim/2,ydim/2]
            amp (float): Scales the height of the terrain.
            low, high (float): The frequencies that wavelets can take on
            num_rough, num_smooth (int): The number of rough/smooth patches
            roughness, smoothness (float): Prominence (radius) of each rough/smooth patch.
            protect (float*float | np.ndarray | list of float | None): Where to place a protected patch 
                (NOTE: None -> patch is disabled).
            protect_radius (float): The radius to leave flat around the protected patch
            protect_decay (float): Gets smoother as protect_decay increases, requires >= 0.""" 
        # satisfy subclass
        super().__init__(terrain_unit, applier)
        # initialize constant parameters
        self.xdim, self.ydim = xdim, ydim
        self.amp = amp
        self.low, self.high = low, high
        self.num_rough, self.num_smooth = num_rough, num_smooth
        self.roughness, self.smoothness = roughness, smoothness
        self.protect, self.protect_radius, self.protect_decay = protect, protect_radius, protect_decay

        # initialize randomizable parameters 
        self.randomize(seed)

    def randomize(self, seed : Union[int, None] = None):
        """
        Randomizes the wavelet frequencies and placement of roughing and smoothing spots/nodes.

        Args:
            seed (int | None): The seed for randomization. If None, a random seed is chosen.
        """
        rand = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        self.wavelet_frequencies =  (np.log(1 + rand.random(self.num_rough)**8) / np.log(2)) * (self.high - self.low) + self.low
        self.wave_signs =           np.sign(rand.random(self.num_rough)-0.5)
        self.rough_spots =          (2*rand.random((self.num_rough,2)) - 1) @ np.array([[self.xdim/2,0],[0,self.ydim/2]])
        self.smooth_spots =         (2*rand.random((self.num_smooth,2)) - 1) @ np.array([[self.xdim/2,0],[0,self.ydim/2]])

    def terrain_fn(self, x, y):
        """
        Computes the height of the terrain at the given (x, y) coordinates.

        Args:
            x (float): The x-coordinate of the point to sample.
            y (float): The y-coordinate of the point to sample.

        Returns:
            float: The height of the terrain at the given (x, y) coordinates.
        """
        def roughen(x,y):
            def wavelet(x,y,frequency,center):
                distance =  np.linalg.norm([x,y]-center)
                bonus =     1 + np.log(1 + 1/frequency) if frequency > 1e-6 else 1
                exp =       np.exp( -(((np.pi/self.roughness)*distance)**2) )
                return bonus * exp * np.cos(np.pi*frequency*distance)
            return sum([wavelet(x,y,self.wavelet_frequencies[n],self.rough_spots[n])*self.wave_signs[n] for n in range(self.num_rough)])
        
        def smoothen(x,y):
            return min(1, (min([np.linalg.norm([x,y]-self.smooth_spots[n]) for n in range(self.num_smooth)])/self.smoothness)**2)

        def protect_center(x,y):
            if self.protect is None:
                return 1
            else:
                exp = np.exp((self.protect_radius - np.linalg.norm([x - self.protect[0],y - self.protect[1]]))/self.protect_decay)
                return np.math.pow(min(1,max(0,1 - exp)), self.protect_decay)
         
        return self.amp * smoothen(x,y) * roughen(x,y) * protect_center(x,y)

    def get_region_tags(self, x, y) -> Set[str]:
        """
        A set of implementation-specific tags associated with the position (x,y) on the terrain.

        Possible Tag Values:
            - spawn_assets : tagged if assets should not be spawned at (x,y)

        Params:
            x (float): 
            y (float): 
        """
        tags = set()

        if x**2 + y**2 >= self.protect_radius ** 2:
            tags.add("spawn_assets")

        return tags

class RoadsTerrain(Terrain):
    """Creates a forest with linear walkways of varying widths that cut about.

    Tags:
        is_road: If the area is part of a road.
        is_spawn: If the area is part of the spawn location.
        """
    def __init__(self, 
                 terrain_unit : float, 
                 applier : Union[Callable[[str],None],None],
                 amp : float = 1.0,
                 xdim : int = 100,
                 ydim : int =100,
                 road_num : int = 6,
                 road_min_width : float = 2,
                 road_max_width : float = 5,
                 spawn_radius : float = 3,
                 border_threshold : float = 2.5,
                 border_height : float = 1.0,
                 bowl_drasticity : int = 4.0,
                 bowl_amplitude : float = 3.0
                 ):
        
        super().__init__(terrain_unit, applier)

        # generation parameters
        self.xdim : int = xdim
        self.ydim : int = ydim
        self.amp : float = amp
        self.road_num : int = road_num
        self.road_min_width : float = road_min_width
        self.road_max_width : float = road_max_width
        self.spawn_radius : float = spawn_radius
        self.border_threshold : float = border_threshold
        self.border_height : float = border_height
        self.bowl_drasticity : int = bowl_drasticity
        self.bowl_amplitude : float = bowl_amplitude

        # randoms (initialized in randomize())
        self.road_widths : Sequence[float] # float \in [road_min_width, road_max_width) list
        self.road_offsets : Sequence[float] # float \in [0,1) list
        self.random_thetas : Sequence[float] # float \in [0,2pi) list
        self.roads : Sequence[Sequence[float]] # float^2 list (float list list)

        self.randomize()

    def randomize(self, seed : Union[int, None] = None):
        """
        Randomizes the wavelet frequencies and placement of roughing and smoothing spots/nodes.

        Args:
            seed (int | None): The seed for randomization. If None, a random seed is chosen.
        """
        rand = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        self.random_thetas = 2*np.pi*rand.random(self.road_num)
        self.roads = [[np.cos(self.random_thetas[n]),
                       np.sin(self.random_thetas[n])] for n in range(self.road_num)]
        self.road_offsets = 0.666 *rand.random(self.road_num) * np.linalg.norm([self.xdim/2 * np.cos(self.random_thetas), self.ydim/2 * np.sin(self.random_thetas)])
        self.road_widths = rand.random(self.road_num) * (self.road_max_width - self.road_min_width) + self.road_min_width

        # Guarantee a path through the origin.
        self.road_offsets[0] = 0
        self.road_widths[0] = self.road_max_width

    # Functions the aid in tagging and making terrain.
    def nearest_road(self, x,y):
        return np.argmin([np.abs(np.dot(self.roads,[x,y])[n] + self.road_offsets[n]) - self.road_widths[n] for n in range(self.road_num)])

    def road(self, x, y):
        nearest_road_id = self.nearest_road(x,y)
        return np.abs(np.dot(self.roads,[x,y])[nearest_road_id] + self.road_offsets[nearest_road_id])
    
    def bowl(self, x, y):
        return self.bowl_amplitude * max(((x**self.bowl_drasticity)/((self.xdim/2))**self.bowl_drasticity),(((y**self.bowl_drasticity)/((self.ydim/2)**self.bowl_drasticity))))
    
    def is_border(self, x, y) -> bool:
        return self.xdim/2 - (np.abs(x) + self.border_threshold) < 0 or self.ydim/2 - (np.abs(y) + self.border_threshold) < 0

    def is_road(self, x:float, y:float) -> bool:
        nearest_road_id = self.nearest_road(x,y)
        return np.abs(np.dot(self.roads,[x,y])[nearest_road_id] + self.road_offsets[nearest_road_id]) < self.road_widths[nearest_road_id]

    def is_spawn(self, x, y):
        return np.linalg.norm([x,y]) < self.spawn_radius
    
    def terrain_fn(self, x, y) -> float:
        return self.amp * self.bowl(x,y) if (self.is_road(x,y) or self.is_spawn(x,y)) and not self.is_border(x,y) else self.amp * (1 + self.bowl(x,y)) + (self.border_height if self.is_border(x,y) else 0)
    
    def get_region_tags(self, x, y) -> Set[str]:
        """
        A set of implementation-specific tags associated with the position (x,y) on the terrain.

        Possible Tag Values:
            - is_road: Tagged if the point is a road.
            - is_spawn: Tagged if the point is inside the spawn area.
        """
        tags = set()

        if self.is_road(x,y) and not self.is_border(x,y):
            tags.add("is_road")

        if self.is_spawn(x,y):
            tags.add("is_spawn")

        return tags
        

        
