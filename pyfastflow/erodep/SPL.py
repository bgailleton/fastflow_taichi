"""
Stream Power Law (SPL) erosion model implementation for landscape evolution.

This module implements the Stream Power Law erosion model using GPU-accelerated
Taichi kernels. The SPL model describes bedrock erosion rate as a function of
drainage area (discharge proxy) and local slope, following the equation:

E = K * A^m * S^n

where:
- E: erosion rate
- K: erodibility coefficient 
- A: drainage area (or discharge)
- S: local slope
- m, n: empirical exponents

The implementation uses an implicit finite difference scheme for numerical
stability, allowing larger time steps while maintaining accuracy.

Key Features:
- Implicit SPL erosion with stability guarantees
- Block uplift implementation
- Integration with FastFlow routing system
- GPU-accelerated computation using Taichi
- Supports landscape evolution modeling

Author: B.G.
"""

import taichi as ti
import numpy as np
import pyfastflow as pf
import math

# Global constants for SPL computation
# Note: These should be moved to constants module in future versions
DT = 1e4      # Time step (years) - TEMPORARY CONSTANT FOR DEVELOPMENT
MEXP = .45    # Drainage area exponent (m in SPL equation)
NEXP = 1.     # Slope exponent (n in SPL equation) - currently unused


@ti.kernel
def block_uplift(z:ti.template(), rate:ti.f32):
	"""
	Apply uniform block uplift to the topography.
	
	Adds vertical motion to all interior nodes (excluding boundary nodes
	that can drain out of the domain). This simulates tectonic uplift
	or other large-scale vertical motions.
	
	Args:
		z (ti.template): Topographic elevation field to be uplifted
		rate (ti.f32): Uplift rate in m/year (positive values = uplift)
	
	Note:
		Boundary nodes that can leave the domain are excluded from uplift
		to maintain consistent boundary conditions.
	
	Author: B.G.
	"""
	for i in z:
		# Only apply uplift to interior nodes (not boundary outlets)
		if(pf.flow.neighbourer_flat.can_leave_domain(i) == False):
			z[i] += rate * DT  # Apply uplift over time step


@ti.kernel
def ext_uplift_nobl(z:ti.template(), rate:ti.template()):
	"""
	
	Author: B.G.
	"""
	for i in z:
		z[i] += rate[i] * DT  # Apply uplift over time step


@ti.kernel
def ext_uplift_bl(z:ti.template(), rate:ti.template()):
	"""
	
	Author: B.G.
	"""
	for i in z:
		# Only apply uplift to interior nodes (not boundary outlets)
		if(pf.flow.neighbourer_flat.can_leave_domain(i) == False):
			z[i] += rate[i] * DT  # Apply uplift over time step


@ti.kernel
def init_erode_SPL(z:ti.template(), z_:ti.template(), z__:ti.template(), alpha_:ti.template(), alpha__:ti.template(), Q:ti.template(), Kr:ti.f32):
	"""
	Initialize fields for implicit Stream Power Law erosion computation.
	
	Sets up the implicit finite difference scheme for SPL erosion by computing
	initial values for the erosion coefficient (alpha) and adjusted elevations.
	The implicit scheme allows stable computation with large time steps.
	
	The erosion follows: dz/dt = -K * Q^m * |∇z|^n
	where K = Kr * Q^m * dt/dx (dimensionless erosion coefficient)
	
	Args:
		z (ti.template): Current topographic elevation field
		z_ (ti.template): Working elevation field for iterations
		z__ (ti.template): Secondary working elevation field
		alpha_ (ti.template): Primary erosion coefficient field
		alpha__ (ti.template): Secondary erosion coefficient field
		Q (ti.template): Discharge field (proxy for drainage area)
		Kr (ti.f32): Base erodibility coefficient (m^(1-2m)/year)
	
	Note:
		The implicit formulation: z_new = (z_old + K*z_downstream)/(1+K)
		results in: alpha = K/(1+K) and z_adjusted = z/(1+K)
	
	Author: B.G.
	"""
	for i in z:
		if(pf.flow.neighbourer_flat.can_leave_domain(i)):
			continue
		# Compute dimensionless erosion coefficient
		# K = erodibility * discharge^m * timestep / grid_spacing
		K = Kr * Q[i]**MEXP * DT / pf.constants.DX
		
		# Implicit scheme coefficients
		alpha_[i] = K / (1 + K)      # Erosion weight factor
		z_[i] = z[i] / (1 + K)       # Adjusted elevation (working field)
		z__[i] = z[i] / (1 + K)      # Adjusted elevation (secondary field)


@ti.kernel
def iteration_erode_SPL(z_:ti.template(), z__:ti.template(), 
	rec_:ti.template(), rec__:ti.template(), 
	alpha_:ti.template(), alpha__:ti.template()):
	"""
	Perform one iteration of implicit SPL erosion using receiver chains.
	
	This kernel implements the core iteration step of the implicit SPL solver.
	It follows drainage paths (receiver chains) to propagate erosion effects
	upstream, ensuring mass conservation and numerical stability.
	
	The algorithm performs parallel graph traversal along flow paths,
	updating elevations based on downstream erosion potential. Multiple
	iterations are needed to fully propagate effects through the drainage network.
	
	Args:
		z_ (ti.template): Working elevation field being updated
		z__ (ti.template): Secondary elevation field for stable updates
		rec_ (ti.template): Primary receiver field (flow routing)
		rec__ (ti.template): Secondary receiver field for stable updates
		alpha_ (ti.template): Primary erosion coefficient field
		alpha__ (ti.template): Secondary erosion coefficient field
	
	Note:
		This implements the implicit update: z_new[i] += alpha[i] * z[receiver[i]]
		where alpha represents the erosion coupling strength between connected nodes.
	
	Author: B.G.
	"""
	# First pass: Update elevations and propagate coefficients downstream
	for i in rec_:
		# Add contribution from downstream node (receiver)
		z_[i] += alpha_[i] * z__[rec_[i]]
		
		# Propagate erosion coefficient through receiver chain
		# This compounds erosion effects along flow paths
		alpha__[i] = alpha_[i] * alpha_[rec_[i]]
		
		# Advance receiver chain by one step (follow flow path)
		rec__[i] = rec_[rec_[i]]

	# Second pass: Update working fields for next iteration
	for i in rec_:
		if(pf.flow.neighbourer_flat.can_leave_domain(i)):
			continue
		z__[i] = z_[i]        # Copy updated elevations
		rec_[i] = rec__[i]    # Advance receiver chains
		alpha_[i] = alpha__[i] # Update erosion coefficients



def SPL(router, alpha_, alpha__, Kr):
	"""
	Execute complete Stream Power Law erosion for one time step.
	
	This function orchestrates the full SPL erosion computation using an implicit
	finite difference scheme. It initializes erosion coefficients, then performs
	iterative updates to propagate erosion effects throughout the drainage network.
	
	The number of iterations is set to log2(N) where N is the total number of nodes,
	which ensures that erosion effects can propagate across the entire domain through
	the longest possible drainage paths.
	
	Args:
		router: FastFlow router object containing topography and flow routing
		alpha_ (ti.field): Primary erosion coefficient field (ti.f32, shape=(N,))
		alpha__ (ti.field): Secondary erosion coefficient field (ti.f32, shape=(N,))
		Kr (float): Base erodibility coefficient (m^(1-2m)/year)
	
	Note:
		The router object is modified in-place, with final eroded topography
		stored in router.z. The function requires pre-computed discharge (router.Q)
		and flow routing (router.receivers_).
	
	Example:
		# Setup erosion coefficient fields
		alpha_ = ti.field(ti.f32, shape=(nx*ny,))
		alpha__ = ti.field(ti.f32, shape=(nx*ny,))
		
		# Run one erosion time step
		SPL(router, alpha_, alpha__, Kr=1e-5)
	
	Author: B.G.
	"""
	# Calculate number of iterations needed for domain-wide propagation
	# log2(N) ensures erosion can propagate through longest drainage paths
	log2 = math.ceil(math.log2(router.nx * router.ny))

	# Initialize erosion coefficients and adjusted elevations
	init_erode_SPL(router.z, router.z_, router.z__, alpha_, alpha__, router.Q, Kr)
	
	# Setup receiver chains for efficient parallel traversal
	router.rec2rec_(second=True)
	
	# Perform iterative erosion computation
	# Each iteration propagates erosion effects one step further upstream
	for _ in range(log2):
		iteration_erode_SPL(router.z_, router.z__, 
						   router.receivers_, router.receivers__, 
						   alpha_, alpha__)

	# Copy final eroded topography back to main elevation field
	router.z.copy_from(router.z_)



@ti.kernel
def erosion_to_source(z:ti.template(), z_:ti.template(), Qs:ti.template()):
	"""
	Convert erosion volumes to sediment source terms.
	
	Calculates the volume of sediment produced by erosion at each node
	by comparing pre- and post-erosion elevations. This can be used as
	a source term for sediment transport models.
	
	Args:
		z (ti.template): Original elevation field (before erosion)
		z_ (ti.template): Eroded elevation field (after erosion)
		Qs (ti.template): Sediment source field to be populated (m³/timestep)
	
	Note:
		Positive Qs values indicate sediment production (erosion).
		The volume calculation assumes uniform grid spacing (DX²).
	
	Author: B.G.
	"""
	for i in z:
		# Calculate eroded volume per cell
		# Volume = elevation_change * cell_area
		Qs[i] = (z[i] - z_[i]) * pf.constants.DX**2/DT

@ti.kernel
def iterate_deposition(z_:ti.template(), Qs:ti.template(), Q:ti.template(), kd:ti.template()):
	"""
	Apply sediment deposition to topography based on transport capacity.
	
	Calculates and applies deposition at each node based on local sediment flux
	and transport capacity. Deposition occurs when sediment supply exceeds the
	transport capacity of the flow.
	
	The deposition amount is limited by:
	1. Transport capacity: kd * Qs[i]/Q[i] (capacity-limited deposition)
	2. Available sediment: Qs[i]/(DX²) (supply-limited deposition)
	3. Non-negative constraint (no negative deposition)
	
	Args:
		z_ (ti.template): Working elevation field to be modified by deposition
		Qs (ti.template): Sediment flux field (m³/timestep)
		Q (ti.template): Water discharge field (m³/s, proxy for transport capacity)
		kd (ti.template): Deposition coefficient field (dimensionless)
	
	Note:
		Deposition is applied only to interior nodes (boundary outlets excluded).
		The formula combines transport capacity and sediment availability constraints.
	
	Author: B.G.
	"""
	for i in z_:
		if(pf.flow.neighbourer_flat.can_leave_domain(i)):
			continue
		
		# Calculate deposition amount based on transport capacity and sediment flux
		# Limited by both transport capacity and available sediment
		deposition = ti.math.max(
			ti.math.min(
				kd * Qs[i]/Q[i],        # Transport capacity limit
				Qs[i]/(pf.constants.DX**2)  # Available sediment limit
			), 
			0.  # Non-negative constraint
		)
		z_[i] += deposition * DT



def SPL_transport(router, alpha_, alpha__, Qs, kr, kd, Nit = 1):
	"""
	Execute complete transport-limited SPL model with erosion and deposition.
	
	Implements the full transport-limited Stream Power Law model that couples
	erosion, sediment transport, and deposition processes. The model iterates
	between erosion computation, sediment flux routing, and deposition to
	achieve equilibrium between erosion and transport capacity.
	
	Workflow for each iteration:
	1. Initialize erosion coefficients based on discharge
	2. Compute erosion using implicit SPL solver
	3. Convert erosion volumes to sediment sources
	4. Route sediment downstream along flow paths
	5. Apply deposition where transport capacity is exceeded
	
	Args:
		router: FastFlow router object with topography and flow routing
		alpha_ (ti.field): Primary erosion coefficient field (ti.f32)
		alpha__ (ti.field): Secondary erosion coefficient field (ti.f32)
		Qs (ti.field): Sediment flux field (m³/timestep)
		kr (float): Base erodibility coefficient (m^(1-2m)/year)
		kd (float or ti.field): Deposition coefficient (dimensionless)
		Nit (int): Number of erosion-transport-deposition iterations (default: 1)
	
	Note:
		Multiple iterations (Nit > 1) allow the system to approach equilibrium
		between erosion and deposition within a single time step. This is
		particularly important for transport-limited conditions.
	
	Example:
		# Setup sediment flux field
		Qs = ti.field(ti.f32, shape=(nx*ny,))
		
		# Run transport-limited erosion
		SPL_transport(router, alpha_, alpha__, Qs, kr=1e-5, kd=1e-2, Nit=5)
	
	Author: B.G.
	"""
	# Calculate number of iterations for erosion solver
	log2 = math.ceil(math.log2(router.nx * router.ny))

	# Initialize erosion coefficients and adjusted elevations once per time step
	init_erode_SPL(router.z, router.z_, router.z__, alpha_, alpha__, router.Q, kr)

	# Main erosion-transport-deposition iteration loop
	change_max = 10000
	it = 0
	while(change_max > 1e-3):
	# for iteration in range(Nit):
		# Setup receiver chains for efficient parallel traversal
		cp_Z = router.z_.to_numpy()
		router.rec2rec_(second=True)
		
		# Perform iterative erosion computation
		# Each iteration propagates erosion effects one step further upstream
		for _ in range(log2):
			iteration_erode_SPL(router.z_, router.z__, 
							   router.receivers_, router.receivers__, 
							   alpha_, alpha__)

		# Convert erosion volumes to sediment sources
		erosion_to_source(router.z, router.z_, Qs)

		# Route sediment downstream along flow paths
		# This accumulates sediment flux at each node
		router.accumulate_custom_donwstream(Qs)

		# Apply deposition based on transport capacity
		iterate_deposition(router.z_, Qs, router.Q, kd)
		change_max = np.mean(np.abs(cp_Z - router.z_.to_numpy()))
		# print(change_max,end=' | ')
		it+=1
	print("Done in", it)
	# Copy final landscape state back to main elevation field
	router.z.copy_from(router.z_)
