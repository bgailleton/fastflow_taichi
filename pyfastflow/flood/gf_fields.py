"""
Flood field management and data structures for 2D shallow water flow.

This module provides the Flooder class which manages field allocation and 
orchestrates the 2D shallow water flow computation pipeline. It handles
GPU memory allocation using Taichi's field system and provides high-level
interface for flood modeling workflows.

The Flooder class integrates with FastFlow's flow routing system to provide
initial conditions and boundary handling for hydrodynamic simulations.
Uses Manning's equation for friction and supports precipitation input.

Key Features:
- GPU memory management for flow depth fields
- Integration with FastFlow flow routing
- Configurable hydrodynamic parameters
- Simple interface for flood simulation workflows

Author: B.G.
"""

import taichi as ti
import pyfastflow as pf
from .. import constants as cte
from . import gf_ls as ls


class Flooder:
	"""
	High-level interface for 2D shallow water flow simulation.
	
	The Flooder class provides a complete workflow for flood modeling using
	GPU-accelerated shallow water equations. It integrates with FastFlow's
	flow routing system and manages field allocation for hydrodynamic variables.
	
	The class handles:
	- GPU memory allocation for flow depth fields
	- Parameter configuration for Manning's friction and precipitation
	- Integration with flow routing for initial conditions
	- Execution of the shallow water flow computation pipeline
	
	Args:
		router: FastFlow router object providing grid and flow routing
		precipitation_rates: Precipitation rate in m/s (default: 10e-3/3600)
		manning: Manning's roughness coefficient (default: 1e-3)
		edge_slope: Boundary slope for edge conditions (default: 1e-2)
	
	Attributes:
		nx (int): Number of grid columns
		ny (int): Number of rows  
		dx (float): Grid cell size
		rshp (tuple): Reshape tuple for converting 1D to 2D arrays
		router: Reference to the FastFlow router
		h (ti.field): Flow depth field
		dh (ti.field): Flow depth change field
		fb (ti.FieldsBuilder): Taichi field builder for memory management
		snodetree: Finalized field structure
	
	Author: B.G.
	"""

	def __init__(self, router, precipitation_rates = 10e-3/3600, manning=0.033, edge_slope = 1e-2, dt_hydro = 1e-3, dt_hydro_ls = None):
		"""
		Initialize the Flooder with grid parameters and hydrodynamic settings using pool-based fields.
		
		Creates a Flooder instance for 2D shallow water flow simulation with automatic
		GPU memory management through the pool system. Allocates all necessary flow
		variables (water depth, discharges) from the pool for efficient computation.
		
		Args:
			router (FlowRouter): FastFlow router object providing grid, elevation data, and flow routing
			precipitation_rates (float, optional): Effective precipitation rate in m/s. 
				Default: 10e-3/3600 (10 mm/hr converted to m/s)
			manning (float, optional): Manning's roughness coefficient (dimensionless).
				Default: 0.033 (typical for grassed surfaces)
			edge_slope (float, optional): Boundary slope for nodes at domain edges (dimensionless).
				Used for computing flow velocity at outlets. Default: 1e-2 (1% slope)
			dt_hydro (float, optional): Time step for GraphFlood hydrodynamic solver in seconds.
				Default: 1e-3 (1 millisecond)
			dt_hydro_ls (float, optional): Time step for LisFlood explicit solver in seconds.
				If None, uses CFL-limited adaptive time stepping. Default: None
				
		Note:
			- All flow fields are allocated from the pool system for efficient memory management
			- Original elevation is saved to CPU memory for reference
			- Fields are automatically released when Flooder is destroyed
			- Both GraphFlood (implicit) and LisFlood (explicit) time steps can be configured
			
		Author: B.G.
		"""

		self.router = router
		self.grid = router.grid

		# Saving the original elevation (to cpu)
		self.og_z = self.grid.z.field.to_numpy()

		# ====== CORE FLOW COMPUTATION FIELDS ======
		# These fields are required for all flow routing operations
		
		self.h = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		self.dh = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		self.nQ = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		self.qx = None
		self.qy = None

		self.h.field.fill(0.)
		self.dh.field.fill(0.)
		self.nQ.field.fill(0.)
		self.nQ_init = False

		self.verbose = False

		cte.PREC = precipitation_rates
		cte.MANNING = manning
		cte.EDGESW = edge_slope
		cte.DT_HYDRO = dt_hydro
		cte.DT_HYDRO_LS = dt_hydro if dt_hydro_ls is None else dt_hydro_ls
		# cte.RAND_RCV = True


	@property
	def nx(self):
		return self.grid.nx

	@property
	def ny(self):
		return self.grid.ny

	@property
	def dx(self):
		return self.grid.dx

	@property
	def rshp(self):
		return self.grid.rshp



	def run_graphflood(self, N=10, N_stochastic = 4, N_diffuse = 0, temporal_filtering = 0.):
		"""
		Execute GraphFlood implicit shallow water flow simulation with pool-based memory management.
		
		Implements the GraphFlood algorithm (Gailleton et al. 2024) that combines flow routing
		with shallow water dynamics for efficient flood modeling.
		
		Algorithm steps for each iteration:
		1. Add water depth to bed elevation (total water surface elevation)
		2. Compute flow receivers using steepest descent on water surface
		3. Reroute flow through lakes and depressions (carving/filling)
		4. Fill topography to ensure flow connectivity
		5. Accumulate flow with stochastic paths (if N_stochastic > 1)
		6. Diffuse discharge field for smoother flow patterns (if N_diffuse > 0)
		7. Apply GraphFlood shallow water equations with Manning's friction
		8. Apply temporal filtering for stability (if temporal_filtering > 0)
		
		Args:
			N (int, optional): Number of major iterations to perform. Default: 10
			N_stochastic (int, optional): Number of stochastic flow realizations per iteration.
				Default: 4. Higher values provide more flow path diversity.
			N_diffuse (int, optional): Number of discharge diffusion steps per iteration.
				Default: 0. Non-zero values smooth discharge patterns.
			temporal_filtering (float, optional): Temporal filtering coefficient (0-1).
				Default: 0.0. Higher values provide more temporal smoothing between iterations.
				
		Note:
			- Uses pool-allocated temporary fields for efficient memory management
			- All temporary fields are automatically released after computation
			- Modifies water depth field (self.h) in place
			- More stable than explicit methods but computationally intensive per iteration
			
		Example:
			flooder.run_graphflood(N=20, N_stochastic=6, N_diffuse=2, temporal_filtering=0.1)
			water_depth = flooder.get_h()  # Get final water depths
			
		Author: B.G.
		"""
		z_          = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		Q_          = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
		receivers_  = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )
		receivers__ = pf.pool.taipool.get_tpfield(dtype = ti.i32, shape = (self.nx*self.ny) )

		for _ in range(N):

			if(self.verbose):
				print('Running iteration', _)

			pf.general_algorithms.util_taichi.add_B_to_A(self.grid.z.field, self.h.field)

			# Compute steepest descent receivers for flow routing
			self.router.compute_receivers()
			
			# Handle flow routing through lakes and depressions
			self.router.reroute_flow()

			# fills with water
			pf.flow.fill_z_add_delta(self.grid.z.field,self.h.field, z_.field,self.router.receivers.field, receivers_.field, receivers__.field, epsilon=1e-3)
			
			# Accumulate with N stochastic routes if N_stochastic > 0else normal, accumulation
			if(N_stochastic > 0):
				self.router.accumulate_constant_Q_stochastic(cte.PREC, area = True, N=N_stochastic)
			else:
				self.router.accumulate_constant_Q(cte.PREC, area = True)

			# Diffuse as multiple flow N times
			for __ in range(N_diffuse):
				pf.flood.diffuse_Q_constant_prec(self.grid.z.field, self.router.Q.field, Q_.field)

			# z is filled with h, I wanna remove the wxtra z
			pf.general_algorithms.util_taichi.add_B_to_weighted_A(self.grid.z.field, self.h.field,-1.)


			if(temporal_filtering == 0.):
				# Apply shallow water equations with Manning's friction
				pf.flood.graphflood_core_cte_mannings(self.h.field, self.grid.z.field, self.dh.field, self.router.receivers.field, self.router.Q.field)
			else:
				if(self.nQ_init == False):
					self.nQ.field.copy_from(self.router.Q.field)
				else:
					pf.general_algorithms.util_taichi.weighted_mean_B_in_A(self.nQ.field, self.router.Q.field, temporal_filtering)

				pf.flood.graphflood_core_cte_mannings(self.h.field, self.grid.z.field, self.dh.field, self.router.receivers.field, self.nQ.field)

		z_.release()
		Q_.release()
		receivers_.release()
		receivers__.release()



	def run_LS(self, N=1000, input_mode = 'constant_prec', mode = None):
		"""
		Execute LisFlood explicit shallow water flow simulation using Bates et al. 2010 method.
		
		Implements the local inertial approximation of the 2D shallow water equations
		with explicit time stepping. The algorithm alternates between flow routing
		(computing unit discharges qx, qy) and depth update (applying continuity equation)
		for N time steps with CFL-limited stability.
		
		Algorithm workflow:
		1. Initialize x/y discharge fields (qx, qy) if not already created
		2. For each time step:
		   a. Apply precipitation input based on input_mode
		   b. Compute flow routing with Manning's friction (flow_route kernel)
		   c. Update water depths using continuity equation (depth_update kernel)
		   d. Apply CFL time step limiting for numerical stability
		
		Args:
			N (int, optional): Number of LisFlood time steps to perform. Default: 1000
			input_mode (str, optional): Precipitation input mode. Default: 'constant_prec'
				- 'constant_prec': Use constant precipitation rate from self.precipitation_rates
				- Other modes may be available depending on implementation
			mode (str, optional): Additional simulation mode (currently unused). Default: None
				
		Note:
			- Uses separate bed elevation (router.z) and water depth (self.h) fields
			- Maintains bed elevation constant, only updates water depth
			- Creates qx, qy discharge fields on first call (pool-allocated)
			- More explicit and faster per time step than GraphFlood
			- Time step size limited by CFL condition for stability
			- Dedicated discharge fields improve numerical accuracy
			
		Example:
			flooder.run_LS(N=5000, input_mode='constant_prec')
			water_depth = flooder.get_h()  # Get final water depths
			velocities_x = flooder.get_qx()  # Get x-direction unit discharges
			
		Author: B.G.
		"""

		# Initialize LisFlood-specific discharge fields on first call
		if self.qx is None :
			# Create x-direction discharge field
			self.qx = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
			self.qy = pf.pool.taipool.get_tpfield(dtype = ti.f32, shape = (self.nx*self.ny) )
			
			self.qx.field.fill(0.)
			self.qy.field.fill(0.)

		# Main LisFlood iteration loop
		for _ in range(N):
			
			if input_mode == 'constant_prec':
				# Optional: Add precipitation as water depth increase
				ls.init_LS_on_hw_from_constant_effective_prec(self.h.field, self.grid.z.field)
			elif input_mode == 'custom_func':
				mode()

			# Step 1: Compute discharge based on water surface gradients
			ls.flow_route(self.h.field, self.grid.z.field, self.qx.field, self.qy.field)
			
			# Step 2: Update water depths based on discharge divergence
			ls.depth_update(self.h.field, self.grid.z.field, self.qx.field, self.qy.field)


	def set_h(self, val):
		"""
		Set flow depth field from numpy array.
		
		Args:
			val (numpy.ndarray): Flow depth values to set
		"""
		self.h.field.from_numpy(val.ravel())
		# self.grid.z.from_numpy(self.og_z.ravel())
		# pf.general_algorithms.util_taichi.add_B_to_A(self.grid.z, self.h)

	def get_h(self):
		"""
		Get current flow depth field as numpy array.
		
		Returns:
			numpy.ndarray: Flow depth values reshaped to 2D grid
		"""
		return self.h.field.to_numpy().reshape(self.rshp)

	def get_qx(self):
		"""
		Get current x-direction discharge field as numpy array.
		
		Returns:
			numpy.ndarray: X-direction discharge values (m²/s) reshaped to 2D grid
		"""
		return self.qx.field.to_numpy().reshape(self.rshp)


	def get_qy(self):
		"""
		Get current y-direction discharge field as numpy array.
		
		Returns:
			numpy.ndarray: Y-direction discharge values (m²/s) reshaped to 2D grid
		"""
		return self.qy.field.to_numpy().reshape(self.rshp)


	def get_dh(self):
		"""
		Get current depth change field as numpy array.
		
		Returns:
			numpy.ndarray: Flow depth change values (m) reshaped to 2D grid
		"""
		return self.dh.field.to_numpy().reshape(self.rshp)


	def get_Q(self):
		"""
		Get current discharge field from router as numpy array.
		
		Returns:
			numpy.ndarray: Discharge values (m³/s) reshaped to 2D grid
		"""
		return self.router.Q.field.to_numpy().reshape(self.rshp)

	def destroy(self):
		"""
		Release all pooled fields and free GPU memory.
		
		Should be called when finished with the Flooder to ensure
		proper cleanup of GPU resources. After calling this method,
		the Flooder should not be used.
		
		Author: B.G.
		"""
		# Release core flood fields back to pool
		if hasattr(self, 'h') and self.h is not None:
			self.h.release()
			self.h = None
			
		if hasattr(self, 'dh') and self.dh is not None:
			self.dh.release()
			self.dh = None
			
		if hasattr(self, 'nQ') and self.nQ is not None:
			self.nQ.release()
			self.nQ = None
			
		# Release LisFlood discharge fields if they exist
		if hasattr(self, 'qx') and self.qx is not None:
			self.qx.release()
			self.qx = None
			
		if hasattr(self, 'qy') and self.qy is not None:
			self.qy.release()
			self.qy = None

	def __del__(self):
		"""
		Destructor - automatically release fields when object is deleted.
		
		Author: B.G.
		"""
		try:
			self.destroy()
		except:
			pass  # Ignore errors during destruction
