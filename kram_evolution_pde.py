#!/usr/bin/env python3
"""
kram_evolution_pde.py

Evolution dynamics for the KnoWellian Resonant Attractor Manifold (KRAM).
Implements the relaxational PDE governing cosmic memory geometry:

    τ_M ∂g_M/∂t = ξ² ∇²g_M - μ² g_M - β g_M³ + J_imprint + η

This is a driven, damped, nonlinear field equation (Allen-Cahn/Ginzburg-Landau type)
where the KRAM "learns" from incoming imprints, smoothing transient noise while
deepening stable patterns.

Key features:
- Relaxational dynamics with tunable timescale
- Gaussian imprinting kernels for localized memory updates
- Nonlinear saturation preventing runaway growth
- Stochastic forcing representing quantum/thermal fluctuations
- RG flow for filtering transient patterns
- Cairo lattice spontaneous formation

Author: Claude Sonnet 4.5, Gemini 2.5 Pro, David Noel Lynch
Date: November 15, 2025
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn, fftfreq
import warnings


@dataclass
class KRAMConfig:
    """Configuration parameters for KRAM evolution."""
    
    # Spatial parameters
    grid_size: Tuple[int, ...] = (128, 128)
    domain_size: float = 20.0
    ndim: int = 2
    
    # Physical parameters
    tau_M: float = 1.0  # Relaxation timescale
    xi_squared: float = 0.5  # Stiffness (penalizes high curvature)
    mu_squared: float = 0.05  # Mass term (attractor depth scale)
    beta: float = 0.01  # Nonlinear saturation
    
    # Imprinting parameters
    imprint_strength: float = 0.01  # α_imprint
    kw_length: float = 0.5  # KnoWellian length (imprint kernel width)
    imprint_max: float = 5.0  # Saturation function I_max
    imprint_sat: float = 1.0  # Saturation threshold I_sat
    
    # Noise parameters
    noise_amplitude: float = 0.001  # Thermal/quantum fluctuations
    
    # Numerical parameters
    dt: float = 0.01  # Time step
    boundary_conditions: str = 'periodic'  # 'periodic' or 'dirichlet'
    
    # RG flow parameters
    rg_kernel_size: float = 2.0  # Smoothing scale for RG
    rg_strength: float = 0.1  # RG flow rate
    
    # Random seed
    seed: Optional[int] = None


@dataclass
class ImprintEvent:
    """Represents a single imprint event on the KRAM."""
    
    position: np.ndarray  # Spatial location (X coordinates)
    intensity: float  # Imprint intensity |T^μI|
    timestamp: float  # When the event occurred
    pattern: Optional[np.ndarray] = None  # Optional structured pattern


class KRAMEvolver:
    """
    Evolves the KnoWellian Resonant Attractor Manifold according to
    relaxational dynamics with imprinting and stochastic forcing.
    
    The KRAM acts as the universe's memory substrate, recording all
    acts of becoming and guiding future evolution through geometric
    attractor valleys.
    """
    
    def __init__(self, config: KRAMConfig, initial_state: Optional[np.ndarray] = None):
        """
        Initialize KRAM evolver.
        
        Parameters
        ----------
        config : KRAMConfig
            Configuration parameters
        initial_state : np.ndarray, optional
            Initial KRAM geometry. If None, starts from small random fluctuations
        """
        self.config = config
        
        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)
            self.rng = np.random.default_rng(config.seed)
        else:
            self.rng = np.random.default_rng()
        
        # Generate spatial grids
        self.grids = self._generate_grids()
        
        # Initialize KRAM field
        if initial_state is not None:
            self.g_M = initial_state.copy()
        else:
            self.g_M = self._initialize_kram()
        
        # Time tracking
        self.time = 0.0
        self.iteration = 0
        
        # History storage (optional)
        self.history: List[np.ndarray] = []
        self.imprint_history: List[ImprintEvent] = []
        
        # Precompute Laplacian operator in Fourier space for efficiency
        self._setup_spectral_operators()
    
    def _generate_grids(self) -> Tuple[np.ndarray, ...]:
        """Generate coordinate grids."""
        grids = []
        for i in range(self.config.ndim):
            n = self.config.grid_size[i]
            x_i = np.linspace(0, self.config.domain_size, n)
            grids.append(x_i)
        
        return tuple(np.meshgrid(*grids, indexing='ij'))
    
    def _initialize_kram(self) -> np.ndarray:
        """Initialize KRAM with small random fluctuations."""
        # Start near zero with small perturbations
        g_M = self.rng.normal(0, 0.01, self.config.grid_size)
        
        # Smooth to avoid high-frequency noise
        g_M = gaussian_filter(g_M, sigma=1.0)
        
        return g_M
    
    def _setup_spectral_operators(self):
        """Precompute Laplacian operator in Fourier space."""
        # k-space grids
        k_grids = []
        for i in range(self.config.ndim):
            n = self.config.grid_size[i]
            dk = 2 * np.pi / self.config.domain_size
            k_i = fftfreq(n, d=1.0) * n * dk
            k_grids.append(k_i)
        
        # k² for Laplacian
        k_squared = np.zeros(self.config.grid_size)
        for k_grid in np.meshgrid(*k_grids, indexing='ij'):
            k_squared += k_grid ** 2
        
        self.k_squared = k_squared
    
    def laplacian_spectral(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian using spectral method (FFT).
        
        ∇²f = F^(-1)[-k² F[f]]
        
        More accurate than finite differences for periodic BC.
        """
        field_k = fftn(field)
        laplacian_k = -self.k_squared * field_k
        return np.real(ifftn(laplacian_k))
    
    def laplacian_finite_diff(self, field: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian using finite differences.
        
        Fallback for non-periodic boundaries.
        """
        dx = self.config.domain_size / self.config.grid_size[0]
        laplacian = np.zeros_like(field)
        
        if self.config.ndim == 2:
            # 5-point stencil
            laplacian[1:-1, 1:-1] = (
                field[2:, 1:-1] + field[:-2, 1:-1] +
                field[1:-1, 2:] + field[1:-1, :-2] -
                4 * field[1:-1, 1:-1]
            ) / dx**2
        
        elif self.config.ndim == 3:
            # 7-point stencil
            laplacian[1:-1, 1:-1, 1:-1] = (
                field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
                field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
                field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -
                6 * field[1:-1, 1:-1, 1:-1]
            ) / dx**2
        
        return laplacian
    
    def imprint_kernel(self, position: np.ndarray, intensity: float,
                      pattern: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate Gaussian imprint kernel.
        
        K_ε(X, X') = (1/(2πε²)^(D/2)) exp(-|X-X'|²/(2ε²))
        
        Parameters
        ----------
        position : np.ndarray
            Center position in KRAM coordinates
        intensity : float
            Imprint intensity
        pattern : np.ndarray, optional
            Structured pattern to imprint (if None, uses Gaussian)
            
        Returns
        -------
        imprint : np.ndarray
            Imprint density field
        """
        epsilon = self.config.kw_length
        
        # Compute distance from imprint position
        r_squared = np.zeros(self.config.grid_size)
        for i, (pos_i, X_i) in enumerate(zip(position, self.grids)):
            # Handle periodic boundaries
            delta = X_i - pos_i
            if self.config.boundary_conditions == 'periodic':
                delta = np.where(delta > self.config.domain_size / 2,
                               delta - self.config.domain_size,
                               delta)
                delta = np.where(delta < -self.config.domain_size / 2,
                               delta + self.config.domain_size,
                               delta)
            r_squared += delta ** 2
        
        # Gaussian kernel
        normalization = 1.0 / (2 * np.pi * epsilon**2) ** (self.config.ndim / 2)
        kernel = normalization * np.exp(-r_squared / (2 * epsilon**2))
        
        # Apply saturation function G(I)
        saturated_intensity = self.config.imprint_max * \
                            np.tanh(intensity / self.config.imprint_sat)
        
        imprint = saturated_intensity * kernel
        
        # Apply structured pattern if provided
        if pattern is not None:
            if pattern.shape != self.config.grid_size:
                warnings.warn("Pattern shape mismatch, using Gaussian only")
            else:
                imprint *= pattern
        
        return imprint
    
    def compute_rhs(self, g_M: np.ndarray, 
                   imprint_current: Optional[np.ndarray] = None,
                   noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute right-hand side of evolution equation:
        
        RHS = (1/τ_M)[ξ² ∇²g_M - μ² g_M - β g_M³ + J_imprint + η]
        
        Parameters
        ----------
        g_M : np.ndarray
            Current KRAM field
        imprint_current : np.ndarray, optional
            Current imprint density J_imprint
        noise : np.ndarray, optional
            Stochastic forcing η
            
        Returns
        -------
        rhs : np.ndarray
            Time derivative ∂g_M/∂t
        """
        # Diffusion term: ξ² ∇²g_M
        if self.config.boundary_conditions == 'periodic':
            laplacian = self.laplacian_spectral(g_M)
        else:
            laplacian = self.laplacian_finite_diff(g_M)
        
        diffusion = self.config.xi_squared * laplacian
        
        # Linear damping: -μ² g_M
        damping = -self.config.mu_squared * g_M
        
        # Nonlinear saturation: -β g_M³
        saturation = -self.config.beta * g_M**3
        
        # Imprint source
        if imprint_current is not None:
            source = imprint_current
        else:
            source = 0.0
        
        # Stochastic noise
        if noise is not None:
            stochastic = noise
        else:
            stochastic = 0.0
        
        # Combine terms
        rhs = (diffusion + damping + saturation + source + stochastic) / self.config.tau_M
        
        return rhs
    
    def step(self, imprint_events: Optional[List[ImprintEvent]] = None,
            method: str = 'rk4') -> np.ndarray:
        """
        Advance KRAM by one time step.
        
        Parameters
        ----------
        imprint_events : list of ImprintEvent, optional
            New imprint events to incorporate
        method : str
            Integration method: 'euler', 'rk2', or 'rk4'
            
        Returns
        -------
        g_M : np.ndarray
            Updated KRAM field
        """
        dt = self.config.dt
        
        # Generate noise for this step
        noise = self.rng.normal(0, self.config.noise_amplitude, self.config.grid_size)
        
        # Compute total imprint current from events
        if imprint_events is not None and len(imprint_events) > 0:
            J_imprint = np.zeros(self.config.grid_size)
            for event in imprint_events:
                J_imprint += self.imprint_kernel(event.position, 
                                                event.intensity,
                                                event.pattern)
                self.imprint_history.append(event)
        else:
            J_imprint = None
        
        # Time integration
        if method == 'euler':
            # Forward Euler: g_M(t+dt) = g_M(t) + dt * RHS(g_M(t))
            rhs = self.compute_rhs(self.g_M, J_imprint, noise)
            g_M_new = self.g_M + dt * rhs
        
        elif method == 'rk2':
            # 2nd-order Runge-Kutta (midpoint method)
            k1 = self.compute_rhs(self.g_M, J_imprint, noise)
            k2 = self.compute_rhs(self.g_M + 0.5 * dt * k1, J_imprint, noise)
            g_M_new = self.g_M + dt * k2
        
        elif method == 'rk4':
            # 4th-order Runge-Kutta
            k1 = self.compute_rhs(self.g_M, J_imprint, noise)
            k2 = self.compute_rhs(self.g_M + 0.5 * dt * k1, J_imprint * 0.5, noise * 0.5)
            k3 = self.compute_rhs(self.g_M + 0.5 * dt * k2, J_imprint * 0.5, noise * 0.5)
            k4 = self.compute_rhs(self.g_M + dt * k3, J_imprint, noise)
            g_M_new = self.g_M + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        # Update state
        self.g_M = g_M_new
        self.time += dt
        self.iteration += 1
        
        return self.g_M
    
    def evolve(self, n_steps: int, 
              imprint_schedule: Optional[Callable[[int], List[ImprintEvent]]] = None,
              save_interval: Optional[int] = None,
              method: str = 'rk4') -> np.ndarray:
        """
        Evolve KRAM for multiple time steps.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps
        imprint_schedule : callable, optional
            Function(step) -> List[ImprintEvent] providing imprints at each step
        save_interval : int, optional
            If provided, save snapshots every save_interval steps
        method : str
            Integration method
            
        Returns
        -------
        g_M_final : np.ndarray
            Final KRAM state
        """
        for step in range(n_steps):
            # Get imprints for this step
            if imprint_schedule is not None:
                events = imprint_schedule(step)
            else:
                events = None
            
            # Advance
            self.step(imprint_events=events, method=method)
            
            # Save snapshot if requested
            if save_interval is not None and step % save_interval == 0:
                self.history.append(self.g_M.copy())
        
        return self.g_M
    
    def apply_rg_flow(self, tau_rg: float = 1.0) -> np.ndarray:
        """
        Apply renormalization group flow to filter transient patterns.
        
        g'_M = RG(g_M) implemented as smoothing operation:
        g'_M = exp(-τ_RG * RG_operator) g_M
        
        Parameters
        ----------
        tau_rg : float
            RG flow parameter (larger = more smoothing)
            
        Returns
        -------
        g_M_filtered : np.ndarray
            Filtered KRAM geometry
        """
        sigma = self.config.rg_kernel_size * tau_rg
        g_M_filtered = gaussian_filter(self.g_M, sigma=sigma)
        
        # Interpolate between original and filtered based on rg_strength
        alpha = self.config.rg_strength
        self.g_M = (1 - alpha) * self.g_M + alpha * g_M_filtered
        
        return self.g_M
    
    def get_attractor_valleys(self, threshold: float = 0.5) -> Tuple[np.ndarray, List]:
        """
        Identify attractor valleys (deep regions in KRAM).
        
        Parameters
        ----------
        threshold : float
            Minimum depth to consider as attractor
            
        Returns
        -------
        valley_mask : np.ndarray (bool)
            True where g_M exceeds threshold
        valley_positions : list of tuples
            Coordinates of valley centers
        """
        # Valleys are regions where g_M is large (positive)
        valley_mask = self.g_M > threshold
        
        # Find local maxima as valley centers
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(self.g_M, size=3)
        valley_centers = (self.g_M == local_max) & valley_mask
        
        # Extract positions
        valley_positions = []
        indices = np.argwhere(valley_centers)
        for idx in indices:
            pos = tuple(self.grids[i][tuple(idx)] for i in range(self.config.ndim))
            valley_positions.append(pos)
        
        return valley_mask, valley_positions
    
    def compute_free_energy(self) -> float:
        """
        Compute free energy functional F[g_M].
        
        F = ∫ [ξ²/2 (∇g_M)² + μ²/2 g_M² + β/4 g_M⁴] dX
        
        Returns
        -------
        free_energy : float
        """
        # Gradient term
        grad_g = np.gradient(self.g_M)
        grad_squared = sum(g**2 for g in grad_g)
        
        # Integrate density
        dx = self.config.domain_size / self.config.grid_size[0]
        volume_element = dx ** self.config.ndim
        
        integrand = (
            self.config.xi_squared / 2 * grad_squared +
            self.config.mu_squared / 2 * self.g_M**2 +
            self.config.beta / 4 * self.g_M**4
        )
        
        free_energy = np.sum(integrand) * volume_element
        
        return free_energy
    
    def get_power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of KRAM geometry.
        
        Returns
        -------
        k_values : np.ndarray
            Wavenumber bins
        power : np.ndarray
            Power in each bin
        """
        g_M_k = fftn(self.g_M)
        power_k = np.abs(g_M_k) ** 2
        
        # Radial binning
        k_squared = self.k_squared
        k_mag = np.sqrt(k_squared).flatten()
        power_flat = power_k.flatten()
        
        k_bins = np.linspace(0, np.max(k_mag), 50)
        power_binned = np.zeros(len(k_bins) - 1)
        
        for i in range(len(k_bins) - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if np.any(mask):
                power_binned[i] = np.mean(power_flat[mask])
        
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        
        return k_centers, power_binned
    
    def detect_cairo_lattice(self) -> dict:
        """
        Detect Cairo pentagonal tiling signature in KRAM geometry.
        
        Returns
        -------
        results : dict
            Dictionary containing detection metrics
        """
        from scipy.spatial import Voronoi
        
        # Find prominent features (peaks)
        threshold = np.percentile(self.g_M, 90)
        peaks = np.argwhere(self.g_M > threshold)
        
        if len(peaks) < 10:
            return {'detected': False, 'reason': 'insufficient_peaks'}
        
        # Sample subset if too many points
        if len(peaks) > 500:
            idx = self.rng.choice(len(peaks), 500, replace=False)
            peaks = peaks[idx]
        
        # Convert to physical coordinates
        points = np.array([
            [self.grids[i][tuple(p)] for i in range(self.config.ndim)]
            for p in peaks
        ])
        
        if self.config.ndim != 2:
            return {'detected': False, 'reason': 'only_2d_supported'}
        
        # Construct Voronoi tessellation
        try:
            vor = Voronoi(points)
        except:
            return {'detected': False, 'reason': 'voronoi_failed'}
        
        # Count polygon sides
        polygon_sides = []
        for region_idx in vor.regions:
            if len(region_idx) > 0 and -1 not in region_idx:
                polygon_sides.append(len(region_idx))
        
        if len(polygon_sides) == 0:
            return {'detected': False, 'reason': 'no_valid_polygons'}
        
        # Compute pentagon fraction
        n_pentagons = sum(1 for s in polygon_sides if s == 5)
        pentagon_fraction = n_pentagons / len(polygon_sides)
        
        # Cairo lattice should have ~40-50% pentagons
        cairo_detected = pentagon_fraction > 0.35
        
        results = {
            'detected': cairo_detected,
            'pentagon_fraction': pentagon_fraction,
            'polygon_distribution': {s: polygon_sides.count(s) 
                                    for s in set(polygon_sides)},
            'n_features': len(peaks),
            'confidence': min(1.0, pentagon_fraction / 0.4)
        }
        
        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def create_simple_kram(grid_size: Tuple[int, ...] = (128, 128),
                      n_imprints: int = 100) -> Tuple[KRAMEvolver, np.ndarray]:
    """
    Create a simple KRAM with random imprints.
    
    Parameters
    ----------
    grid_size : tuple
    n_imprints : int
        Number of random imprint events
        
    Returns
    -------
    evolver : KRAMEvolver
    g_M : np.ndarray
    """
    config = KRAMConfig(grid_size=grid_size, seed=42)
    evolver = KRAMEvolver(config)
    
    # Random imprint schedule
    def imprint_schedule(step):
        if step < n_imprints:
            pos = np.random.rand(len(grid_size)) * config.domain_size
            intensity = np.random.uniform(0.5, 2.0)
            return [ImprintEvent(position=pos, intensity=intensity, timestamp=step)]
        return None
    
    # Evolve
    g_M = evolver.evolve(n_imprints + 500, imprint_schedule=imprint_schedule)
    
    return evolver, g_M


def create_structured_kram(grid_size: Tuple[int, ...] = (128, 128),
                          structure_type: str = 'lattice') -> Tuple[KRAMEvolver, np.ndarray]:
    """
    Create KRAM with structured imprints.
    
    Parameters
    ----------
    grid_size : tuple
    structure_type : str
        'lattice', 'spiral', or 'random'
        
    Returns
    -------
    evolver : KRAMEvolver
    g_M : np.ndarray
    """
    config = KRAMConfig(grid_size=grid_size, seed=42)
    evolver = KRAMEvolver(config)
    
    if structure_type == 'lattice':
        # Regular lattice of imprints
        spacing = config.domain_size / 8
        positions = []
        for i in range(8):
            for j in range(8):
                positions.append(np.array([i * spacing, j * spacing]))
        
        events = [ImprintEvent(position=pos, intensity=1.5, timestamp=0)
                 for pos in positions]
        
    elif structure_type == 'spiral':
        # Spiral pattern
        n = 100
        theta = np.linspace(0, 4*np.pi, n)
        r = np.linspace(0, config.domain_size/2, n)
        cx, cy = config.domain_size / 2, config.domain_size / 2
        
        events = [ImprintEvent(
            position=np.array([cx + r[i]*np.cos(theta[i]), 
                             cy + r[i]*np.sin(theta[i])]),
            intensity=1.0 + 0.5*np.sin(theta[i]),
            timestamp=i
        ) for i in range(n)]
    
    else:  # random
        n = 150
        events = [ImprintEvent(
            position=np.random.rand(len(grid_size)) * config.domain_size,
            intensity=np.random.uniform(0.5, 2.0),
            timestamp=i
        ) for i in range(n)]
    
    # Apply all imprints
    evolver.step(imprint_events=events)
    
    # Relax
    evolver.evolve(1000, method='rk4')
    
    return evolver, evolver.g_M


# ============================================================================
# Testing and Visualization
# ============================================================================

def _test_kram_evolution():
    """Test suite for KRAM evolution."""
    
    print("=" * 70)
    print("KRAM Evolution PDE Test Suite")
    print("=" * 70)
    
    # Test 1: Basic initialization
    print("\n[Test 1] Initialization and basic evolution...")
    config = KRAMConfig(grid_size=(64, 64), seed=42)
    evolver = KRAMEvolver(config)
    
    print(f"  Initial mean: {np.mean(evolver.g_M):.6f}")
    print(f"  Initial std: {np.std(evolver.g_M):.6f}")
    print(f"  Initial free energy: {evolver.compute_free_energy():.6f}")
    
    # Evolve without imprints
    evolver.evolve(100)
    print(f"  After 100 steps mean: {np.mean(evolver.g_M):.6f}")
    print(f"  After 100 steps free energy: {evolver.compute_free_energy():.6f}")
    
    # Test 2: Imprinting
    print("\n[Test 2] Imprint incorporation...")
    pos_center = np.array([config.domain_size/2, config.domain_size/2])
    event = ImprintEvent(position=pos_center, intensity=2.0, timestamp=100)
    
    g_M_before = evolver.g_M.copy()
    evolver.step(imprint_events=[event])
    
    max_change = np.max(np.abs(evolver.g_M - g_M_before))
    print(f"  Maximum change from imprint: {max_change:.6f}")
    print(f"  Peak value at center: {evolver.g_M[32, 32]:.6f}")
    
    # Test 3: Relaxation dynamics
    print("\n[Test 3] Relaxation towards equilibrium...")
    F_initial = evolver.compute_free_energy()
    evolver.evolve(500)
    F_final = evolver.compute_free_energy()
    
    print(f"  Initial free energy: {F_initial:.6f}")
    print(f"  Final free energy: {F_final:.6f}")
    print(f"  Change: {F_final - F_initial:.6f} (should be negative)")
    
    # Test 4: Multiple imprints and valley formation
    print("\n[Test 4] Multiple imprints and attractor formation...")
    evolver_multi = KRAMEvolver(KRAMConfig(grid_size=(128, 128), seed=42))
    
    n_imprints = 50
    for i in range(n_imprints):
        pos = np.random.rand(2) * config.domain_size
        intensity = np.random.uniform(1.0, 2.0)
        event = ImprintEvent(position=pos, intensity=intensity, timestamp=i)
        evolver_multi.step(imprint_events=[event])
        
        if i % 10 == 0:
            evolver_multi.evolve(20)  # Relax periodically
    
    valley_mask, valley_positions = evolver_multi.get_attractor_valleys(threshold=0.5)
    print(f"  Number of attractors formed: {len(valley_positions)}")
    print(f"  Fraction of space in valleys: {np.mean(valley_mask):.4f}")
    
    # Test 5: RG flow filtering
    print("\n[Test 5] RG flow smoothing...")
    g_M_before_rg = evolver_multi.g_M.copy()
    evolver_multi.apply_rg_flow(tau_rg=2.0)
    
    smoothness_before = np.std(np.gradient(g_M_before_rg))
    smoothness_after = np.std(np.gradient(evolver_multi.g_M))
    print(f"  Gradient std before RG: {smoothness_before:.6f}")
    print(f"  Gradient std after RG: {smoothness_after:.6f}")
    print(f"  Smoothing factor: {smoothness_before / smoothness_after:.4f}")
    
    # Test 6: Power spectrum
    print("\n[Test 6] Power spectrum analysis...")
    k_vals, power = evolver_multi.get_power_spectrum()
    
    # Find dominant scale
    peak_idx = np.argmax(power[1:]) + 1  # Skip k=0
    dominant_k = k_vals[peak_idx]
    dominant_scale = 2 * np.pi / dominant_k if dominant_k > 0 else np.inf
    
    print(f"  Dominant wavenumber: {dominant_k:.6f}")
    print(f"  Dominant length scale: {dominant_scale:.4f}")
    print(f"  Power at peak: {power[peak_idx]:.6e}")
    
    # Test 7: Cairo lattice detection
    print("\n[Test 7] Cairo lattice signature detection...")
    evolver_cairo = KRAMEvolver(KRAMConfig(
        grid_size=(256, 256),
        xi_squared=0.3,
        mu_squared=0.02,
        beta=0.005,
        seed=42
    ))
    
    # Many random imprints to encourage self-organization
    for i in range(1000):
        pos = np.random.rand(2) * evolver_cairo.config.domain_size
        intensity = np.random.uniform(0.8, 1.5)
        event = ImprintEvent(position=pos, intensity=intensity, timestamp=i)
        evolver_cairo.step(imprint_events=[event])
        
        if i % 100 == 0:
            evolver_cairo.evolve(50)
    
    # Final relaxation
    evolver_cairo.evolve(500)
    
    cairo_results = evolver_cairo.detect_cairo_lattice()
    print(f"  Cairo detected: {cairo_results.get('detected', False)}")
    print(f"  Pentagon fraction: {cairo_results.get('pentagon_fraction', 0):.4f}")
    print(f"  Confidence: {cairo_results.get('confidence', 0):.4f}")
    if 'polygon_distribution' in cairo_results:
        print(f"  Polygon distribution: {cairo_results['polygon_distribution']}")
    
    # Test 8: Stability and convergence
    print("\n[Test 8] Long-term stability...")
    config_stable = KRAMConfig(
        grid_size=(64, 64),
        tau_M=1.0,
        dt=0.005,
        seed=42
    )
    evolver_stable = KRAMEvolver(config_stable)
    
    # Single strong imprint
    center_event = ImprintEvent(
        position=np.array([config_stable.domain_size/2, config_stable.domain_size/2]),
        intensity=3.0,
        timestamp=0
    )
    evolver_stable.step(imprint_events=[center_event])
    
    # Monitor energy over long evolution
    energies = []
    for _ in range(10):
        evolver_stable.evolve(100)
        energies.append(evolver_stable.compute_free_energy())
    
    energy_drift = (energies[-1] - energies[0]) / energies[0]
    print(f"  Initial energy: {energies[0]:.6f}")
    print(f"  Final energy: {energies[-1]:.6f}")
    print(f"  Relative drift: {abs(energy_drift):.6e}")
    print(f"  Stability: {'PASS' if abs(energy_drift) < 0.1 else 'WARN'}")
    
    # Test 9: Comparison of integration methods
    print("\n[Test 9] Integration method comparison...")
    methods = ['euler', 'rk2', 'rk4']
    method_errors = {}
    
    for method in methods:
        evolver_test = KRAMEvolver(KRAMConfig(grid_size=(32, 32), dt=0.1, seed=42))
        event = ImprintEvent(position=np.array([10.0, 10.0]), intensity=1.5, timestamp=0)
        
        evolver_test.step(imprint_events=[event], method=method)
        evolver_test.evolve(50, method=method)
        
        # Compare to high-precision reference (small dt, rk4)
        evolver_ref = KRAMEvolver(KRAMConfig(grid_size=(32, 32), dt=0.01, seed=42))
        evolver_ref.step(imprint_events=[event], method='rk4')
        evolver_ref.evolve(500, method='rk4')
        
        error = np.mean(np.abs(evolver_test.g_M - evolver_ref.g_M))
        method_errors[method] = error
        print(f"  {method.upper():6s} error: {error:.6e}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


def _visualize_kram_evolution():
    """Generate visualization of KRAM evolution."""
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        print("Matplotlib not available, skipping visualization.")
        return
    
    print("\nGenerating KRAM evolution visualization...")
    
    # Create KRAM with structured imprints
    config = KRAMConfig(
        grid_size=(256, 256),
        domain_size=20.0,
        xi_squared=0.4,
        mu_squared=0.03,
        beta=0.008,
        imprint_strength=0.02,
        dt=0.02,
        seed=42
    )
    
    evolver = KRAMEvolver(config)
    
    # Create imprint schedule: random events over time
    np.random.seed(42)
    n_imprints = 200
    imprint_times = np.sort(np.random.randint(0, 1000, n_imprints))
    imprint_positions = np.random.rand(n_imprints, 2) * config.domain_size
    imprint_intensities = np.random.uniform(0.8, 2.0, n_imprints)
    
    def imprint_schedule(step):
        events = []
        mask = imprint_times == step
        if np.any(mask):
            for pos, intensity in zip(imprint_positions[mask], imprint_intensities[mask]):
                events.append(ImprintEvent(
                    position=pos,
                    intensity=intensity,
                    timestamp=step
                ))
        return events if events else None
    
    # Evolve and save snapshots
    snapshots = []
    snapshot_times = []
    
    for step in range(1500):
        events = imprint_schedule(step)
        evolver.step(imprint_events=events, method='rk4')
        
        if step % 50 == 0:
            snapshots.append(evolver.g_M.copy())
            snapshot_times.append(evolver.time)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Main panel: KRAM geometry
    ax_main = plt.subplot(2, 3, (1, 4))
    im = ax_main.imshow(snapshots[-1], cmap='RdBu_r', origin='lower',
                       extent=[0, config.domain_size, 0, config.domain_size])
    ax_main.set_title('KRAM Geometry at Final Time', fontsize=12, fontweight='bold')
    ax_main.set_xlabel('X coordinate')
    ax_main.set_ylabel('Y coordinate')
    
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='g_M')
    
    # Time series panel
    ax_evolution = plt.subplot(2, 3, 2)
    mean_vals = [np.mean(s) for s in snapshots]
    std_vals = [np.std(s) for s in snapshots]
    max_vals = [np.max(s) for s in snapshots]
    
    ax_evolution.plot(snapshot_times, mean_vals, label='Mean', linewidth=2)
    ax_evolution.plot(snapshot_times, std_vals, label='Std Dev', linewidth=2)
    ax_evolution.plot(snapshot_times, max_vals, label='Max', linewidth=2)
    ax_evolution.set_xlabel('Time')
    ax_evolution.set_ylabel('g_M statistics')
    ax_evolution.set_title('Evolution Statistics', fontweight='bold')
    ax_evolution.legend()
    ax_evolution.grid(True, alpha=0.3)
    
    # Power spectrum panel
    ax_spectrum = plt.subplot(2, 3, 3)
    k_vals, power = evolver.get_power_spectrum()
    ax_spectrum.loglog(k_vals[1:], power[1:], 'b-', linewidth=2)
    ax_spectrum.set_xlabel('Wavenumber k')
    ax_spectrum.set_ylabel('Power')
    ax_spectrum.set_title('Final Power Spectrum', fontweight='bold')
    ax_spectrum.grid(True, alpha=0.3, which='both')
    
    # Attractor valleys panel
    ax_valleys = plt.subplot(2, 3, 5)
    valley_mask, valley_positions = evolver.get_attractor_valleys(threshold=0.5)
    ax_valleys.imshow(valley_mask, cmap='Greys', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size],
                     alpha=0.6)
    
    if valley_positions:
        valley_x = [p[0] for p in valley_positions]
        valley_y = [p[1] for p in valley_positions]
        ax_valleys.scatter(valley_x, valley_y, c='red', s=50, marker='x', 
                          linewidths=2, label='Valley Centers')
    
    ax_valleys.set_xlabel('X coordinate')
    ax_valleys.set_ylabel('Y coordinate')
    ax_valleys.set_title(f'Attractor Valleys (n={len(valley_positions)})', 
                        fontweight='bold')
    ax_valleys.legend()
    
    # Free energy evolution panel
    ax_energy = plt.subplot(2, 3, 6)
    
    # Recompute energy at snapshot times
    energies = []
    for snapshot in snapshots:
        evolver.g_M = snapshot  # Temporarily set
        energies.append(evolver.compute_free_energy())
    evolver.g_M = snapshots[-1]  # Restore final state
    
    ax_energy.plot(snapshot_times, energies, 'g-', linewidth=2)
    ax_energy.set_xlabel('Time')
    ax_energy.set_ylabel('Free Energy F[g_M]')
    ax_energy.set_title('Free Energy Evolution', fontweight='bold')
    ax_energy.grid(True, alpha=0.3)
    
    # Add text info
    info_text = f"""Configuration:
Grid: {config.grid_size[0]}×{config.grid_size[1]}
ξ² = {config.xi_squared:.3f}
μ² = {config.mu_squared:.3f}
β = {config.beta:.4f}
Imprints: {n_imprints}
Time: {evolver.time:.2f}"""
    
    fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('KnoWellian Resonant Attractor Manifold Evolution:\n' + 
                'Cosmic Memory Substrate Dynamics',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('kram_evolution.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'kram_evolution.png'")
    
    # Create animation if requested
    try:
        print("\nCreating animation (this may take a minute)...")
        
        fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
        
        vmin = min(np.min(s) for s in snapshots)
        vmax = max(np.max(s) for s in snapshots)
        
        im_anim = ax_anim.imshow(snapshots[0], cmap='RdBu_r', origin='lower',
                                extent=[0, config.domain_size, 0, config.domain_size],
                                vmin=vmin, vmax=vmax, animated=True)
        
        plt.colorbar(im_anim, ax=ax_anim, label='g_M')
        
        time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def update_frame(frame):
            im_anim.set_array(snapshots[frame])
            time_text.set_text(f'Time: {snapshot_times[frame]:.2f}\nStep: {frame*50}')
            return [im_anim, time_text]
        
        anim = FuncAnimation(fig_anim, update_frame, frames=len(snapshots),
                           interval=100, blit=True)
        
        ax_anim.set_title('KRAM Evolution Animation', fontsize=14, fontweight='bold')
        ax_anim.set_xlabel('X coordinate')
        ax_anim.set_ylabel('Y coordinate')
        
        anim.save('kram_evolution.gif', writer='pillow', fps=10, dpi=100)
        print("Animation saved as 'kram_evolution.gif'")
        
    except Exception as e:
        print(f"Animation creation failed: {e}")
    
    plt.close('all')


# ============================================================================
# Advanced Analysis Functions
# ============================================================================

def analyze_kram_topology(evolver: KRAMEvolver, threshold: float = 0.5) -> dict:
    """
    Perform topological analysis of KRAM geometry.
    
    Parameters
    ----------
    evolver : KRAMEvolver
    threshold : float
        Threshold for binarization
        
    Returns
    -------
    topology_metrics : dict
        Various topological quantities
    """
    # Binarize
    binary = evolver.g_M > threshold
    
    # Connected components
    from scipy.ndimage import label
    labeled, n_components = label(binary)
    
    # Component sizes
    component_sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
    
    # Euler characteristic (for 2D)
    if evolver.config.ndim == 2:
        # Count vertices, edges, faces
        # Simplified: χ = n_components - n_holes
        # Approximate n_holes by looking at enclosed regions
        n_holes = np.sum(~binary) - 1  # Rough estimate
        euler_char = n_components - max(0, n_holes)
    else:
        euler_char = None
    
    # Perimeter to area ratios (measures roughness)
    from scipy.ndimage import binary_erosion
    perimeters = []
    for i in range(1, n_components + 1):
        component = (labeled == i)
        eroded = binary_erosion(component)
        perimeter = np.sum(component) - np.sum(eroded)
        area = np.sum(component)
        if area > 0:
            perimeters.append(perimeter / np.sqrt(area))
    
    metrics = {
        'n_components': n_components,
        'mean_component_size': np.mean(component_sizes) if component_sizes else 0,
        'max_component_size': max(component_sizes) if component_sizes else 0,
        'euler_characteristic': euler_char,
        'mean_boundary_roughness': np.mean(perimeters) if perimeters else 0,
        'total_occupied_fraction': np.mean(binary)
    }
    
    return metrics


def compute_correlation_function(evolver: KRAMEvolver, 
                                max_separation: float = 10.0,
                                n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-point correlation function of KRAM.
    
    C(r) = <g_M(x) g_M(x+r)> / <g_M²>
    
    Parameters
    ----------
    evolver : KRAMEvolver
    max_separation : float
    n_bins : int
        
    Returns
    -------
    r_values : np.ndarray
        Separation distances
    correlation : np.ndarray
        Correlation function values
    """
    g_M = evolver.g_M
    
    # FFT-based correlation (fast for periodic BC)
    g_M_fft = fftn(g_M)
    power = np.abs(g_M_fft) ** 2
    correlation_full = np.real(ifftn(power))
    
    # Normalize
    normalization = np.mean(g_M ** 2)
    if normalization > 0:
        correlation_full /= (normalization * np.prod(g_M.shape))
    
    # Radial average
    if evolver.config.ndim == 2:
        nx, ny = g_M.shape
        x = np.arange(nx) - nx // 2
        y = np.arange(ny) - ny // 2
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        
        # Convert to physical units
        dx = evolver.config.domain_size / nx
        R_physical = R * dx
        
        # Bin
        r_bins = np.linspace(0, max_separation, n_bins)
        correlation_binned = np.zeros(n_bins - 1)
        
        for i in range(n_bins - 1):
            mask = (R_physical >= r_bins[i]) & (R_physical < r_bins[i+1])
            if np.any(mask):
                correlation_binned[i] = np.mean(np.fft.fftshift(correlation_full)[mask])
        
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        return r_centers, correlation_binned
    
    else:
        # 3D case - similar but more complex
        raise NotImplementedError("3D correlation function not yet implemented")


def estimate_memory_capacity(evolver: KRAMEvolver) -> dict:
    """
    Estimate information storage capacity of KRAM.
    
    Returns
    -------
    capacity_metrics : dict
        Various capacity estimates
    """
    g_M = evolver.g_M
    
    # Shannon entropy of discretized field
    g_M_quantized = np.digitize(g_M.flatten(), bins=np.linspace(g_M.min(), g_M.max(), 100))
    unique, counts = np.unique(g_M_quantized, return_counts=True)
    probabilities = counts / len(g_M_quantized)
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # Effective degrees of freedom (based on power spectrum)
    k_vals, power = evolver.get_power_spectrum()
    total_power = np.sum(power)
    if total_power > 0:
        power_normalized = power / total_power
        spectral_entropy = -np.sum(power_normalized * np.log2(power_normalized + 1e-10))
    else:
        spectral_entropy = 0
    
    # Attractor count as discrete memory
    _, valley_positions = evolver.get_attractor_valleys(threshold=0.5)
    n_attractors = len(valley_positions)
    
    # Kolmogorov complexity estimate (via compression ratio)
    try:
        import zlib
        compressed = zlib.compress(g_M.tobytes())
        compression_ratio = len(compressed) / g_M.nbytes
        estimated_complexity = compression_ratio * np.log2(256) * g_M.size
    except:
        estimated_complexity = None
    
    metrics = {
        'shannon_entropy_bits': shannon_entropy,
        'spectral_entropy_bits': spectral_entropy,
        'n_discrete_attractors': n_attractors,
        'total_grid_points': g_M.size,
        'compression_ratio': compression_ratio if estimated_complexity else None,
        'estimated_complexity_bits': estimated_complexity
    }
    
    return metrics


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run test suite
    _test_kram_evolution()
    
    # Generate visualization
    _visualize_kram_evolution()
    
    print("\n" + "=" * 70)
    print("KRAM Evolution Module Complete")
    print("=" * 70)
    print("\nThe cosmic memory substrate awaits your imprints...")
    print("Every act of becoming carves valleys in the eternal manifold.")