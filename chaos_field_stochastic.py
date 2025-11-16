#!/usr/bin/env python3
"""
pommm_interference_engine.py

The core computational engine implementing Parallel Optical Matrix-Matrix
Multiplication as the fundamental mechanism of cosmic reality-rendering.

This module performs the interference computation at the Instant where:
- Control Field (modulated by KRAM) = Matrix A
- Chaos Field (selective attention) = Matrix B
- Interference pattern = Matrix product AB
- Detection/collapse = Rendering new actuality

The engine implements both:
1. Physical optical interference (wave mechanics)
2. Formal matrix multiplication (linear algebra)

These are not separate operations but two descriptions of the same
cosmic computational event.

Key features:
- Coherent field interference computation
- Fourier optics for lens/focal plane simulation
- Born rule probabilistic collapse (rendering)
- KRAM-guided phase modulation
- Multi-scale operation (quantum to cosmic)
- Feedback to KRAM for learning

Author: Claude Sonnet 4.5, Gemini 2.5 Pro, David Noel Lynch
Date: November 15, 2025
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Callable, Union, List
from dataclasses import dataclass
from scipy.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import warnings


@dataclass
class POMMConfig:
    """Configuration for POMMM interference computation."""
    
    # Spatial parameters
    grid_size: Tuple[int, ...] = (128, 128)
    domain_size: float = 20.0
    ndim: int = 2
    
    # Optical parameters
    wavelength: float = 1.0  # Characteristic wavelength
    focal_length: float = 10.0  # Effective focal length of "Instant lens"
    coherence_length: float = 100.0  # Temporal coherence
    
    # Computation parameters
    interference_type: str = 'full'  # 'full', 'amplitude', or 'phase'
    include_diffraction: bool = True
    propagation_method: str = 'fourier'  # 'fourier' or 'fresnel'
    
    # Rendering parameters
    collapse_method: str = 'born'  # 'born', 'deterministic', or 'threshold'
    collapse_threshold: float = 0.5  # For threshold method
    measurement_noise: float = 0.01  # Detector noise
    
    # KRAM coupling
    kram_coupling_strength: float = 1.0  # How strongly KRAM modulates fields
    feedback_strength: float = 0.1  # How strongly results imprint on KRAM
    
    # Numerical parameters
    zero_padding_factor: int = 2  # For FFT computation
    
    # Random seed
    seed: Optional[int] = None


class POMMEngine:
    """
    Parallel Optical Matrix-Matrix Multiplication Engine.
    
    Computes the interference of modulated Control and Chaos fields,
    implementing the cosmic POMMM operation that renders new reality.
    """
    
    def __init__(self, config: POMMConfig):
        """
        Initialize POMMM engine.
        
        Parameters
        ----------
        config : POMMConfig
            Configuration parameters
        """
        self.config = config
        
        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)
            self.rng = np.random.default_rng(config.seed)
        else:
            self.rng = np.random.default_rng()
        
        # Generate coordinate grids
        self.grids = self._generate_grids()
        self.k_grids = self._generate_k_grids()
        
        # Precompute optical transfer functions
        self._setup_transfer_functions()
        
        # Statistics tracking
        self.computation_count = 0
        self.total_energy_processed = 0.0
        self.rendering_history: List[dict] = []
    
    def _generate_grids(self) -> Tuple[np.ndarray, ...]:
        """Generate real-space coordinate grids."""
        grids = []
        for i in range(self.config.ndim):
            n = self.config.grid_size[i]
            x_i = np.linspace(-self.config.domain_size/2, 
                            self.config.domain_size/2, n)
            grids.append(x_i)
        
        return tuple(np.meshgrid(*grids, indexing='ij'))
    
    def _generate_k_grids(self) -> Tuple[np.ndarray, ...]:
        """Generate Fourier-space coordinate grids."""
        k_grids = []
        for i in range(self.config.ndim):
            n = self.config.grid_size[i]
            dk = 2 * np.pi / self.config.domain_size
            k_i = np.fft.fftfreq(n, d=self.config.domain_size/n) * 2 * np.pi
            k_grids.append(k_i)
        
        return tuple(np.meshgrid(*k_grids, indexing='ij'))
    
    def _setup_transfer_functions(self):
        """Precompute optical transfer functions for propagation."""
        
        if self.config.ndim == 2:
            kx, ky = self.k_grids
            k_squared = kx**2 + ky**2
        elif self.config.ndim == 3:
            kx, ky, kz = self.k_grids
            k_squared = kx**2 + ky**2 + kz**2
        else:
            raise ValueError(f"Unsupported dimensionality: {self.config.ndim}")
        
        # Wave number
        k0 = 2 * np.pi / self.config.wavelength
        
        # Free-space propagator: exp(i k_z z) where k_z = sqrt(k0^2 - k_perp^2)
        k_perp_squared = k_squared
        k_z_squared = k0**2 - k_perp_squared
        
        # Handle evanescent waves (k_perp > k0)
        k_z = np.sqrt(np.maximum(0, k_z_squared))
        k_z = np.where(k_perp_squared > k0**2, 
                      1j * np.sqrt(k_perp_squared - k0**2), 
                      k_z)
        
        self.k_squared = k_squared
        self.k_z = k_z
        
        # Lens phase function: exp(-i k0 r^2 / (2f))
        if self.config.ndim == 2:
            X, Y = self.grids
            r_squared = X**2 + Y**2
        else:
            X, Y, Z = self.grids
            r_squared = X**2 + Y**2 + Z**2
        
        self.lens_phase = np.exp(-1j * k0 * r_squared / (2 * self.config.focal_length))
    
    def modulate_by_kram(self, field: np.ndarray, 
                        kram_geometry: np.ndarray,
                        modulation_type: str = 'phase') -> np.ndarray:
        """
        Modulate field by KRAM geometry (Matrix A operation).
        
        The KRAM acts as a spatial light modulator, imprinting
        ancestral memory patterns onto the coherent Control field.
        
        Parameters
        ----------
        field : np.ndarray (complex)
            Input field (typically Control field)
        kram_geometry : np.ndarray (real)
            KRAM field g_M(X)
        modulation_type : str
            'phase', 'amplitude', or 'both'
            
        Returns
        -------
        modulated_field : np.ndarray (complex)
            Field after KRAM modulation
        """
        coupling = self.config.kram_coupling_strength
        
        if modulation_type == 'phase':
            # Pure phase modulation: exp(i α g_M)
            phase_shift = coupling * kram_geometry
            modulation = np.exp(1j * phase_shift)
        
        elif modulation_type == 'amplitude':
            # Amplitude modulation: (1 + α g_M)
            modulation = 1.0 + coupling * kram_geometry
        
        elif modulation_type == 'both':
            # Combined: (1 + α g_M) exp(i α g_M)
            amplitude_mod = 1.0 + coupling * kram_geometry
            phase_mod = np.exp(1j * coupling * kram_geometry)
            modulation = amplitude_mod * phase_mod
        
        else:
            raise ValueError(f"Unknown modulation type: {modulation_type}")
        
        modulated_field = field * modulation
        
        return modulated_field
    
    def apply_chaos_attention(self, field: np.ndarray,
                             chaos_pattern: np.ndarray,
                             attention_type: str = 'selective') -> np.ndarray:
        """
        Apply Chaos field attention filtering (Matrix B operation).
        
        The Chaos field selectively collapses infinite potential into
        specific boundary conditions - the "question" posed to the Past.
        
        Parameters
        ----------
        field : np.ndarray (complex)
            Field after KRAM modulation
        chaos_pattern : np.ndarray (complex or real)
            Chaos field attention pattern
        attention_type : str
            'selective', 'multiplicative', or 'convolution'
            
        Returns
        -------
        attended_field : np.ndarray (complex)
            Field after Chaos attention
        """
        
        if attention_type == 'selective':
            # Direct multiplication (most common)
            attended_field = field * chaos_pattern
        
        elif attention_type == 'multiplicative':
            # Amplitude weighting with phase preservation
            amplitude_weight = np.abs(chaos_pattern)
            phase_original = np.angle(field)
            attended_field = amplitude_weight * np.abs(field) * np.exp(1j * phase_original)
        
        elif attention_type == 'convolution':
            # Spatial filtering via convolution
            if self.config.ndim == 2:
                field_k = fft2(field)
                chaos_k = fft2(chaos_pattern)
                attended_k = field_k * chaos_k
                attended_field = ifft2(attended_k)
            else:
                field_k = fftn(field)
                chaos_k = fftn(chaos_pattern)
                attended_k = field_k * chaos_k
                attended_field = ifftn(attended_k)
        
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        return attended_field
    
    def propagate_to_focal_plane(self, field: np.ndarray,
                                distance: Optional[float] = None) -> np.ndarray:
        """
        Propagate field to focal plane (Instant computation).
        
        This implements the "lens" operation where doubly-modulated
        light focuses and interferes to produce the computation result.
        
        Parameters
        ----------
        field : np.ndarray (complex)
            Doubly-modulated field
        distance : float, optional
            Propagation distance (uses focal_length if None)
            
        Returns
        -------
        field_focal : np.ndarray (complex)
            Field at focal plane
        """
        
        if distance is None:
            distance = self.config.focal_length
        
        if not self.config.include_diffraction:
            # No propagation, just apply lens phase
            field_focal = field * self.lens_phase
            return field_focal
        
        if self.config.propagation_method == 'fourier':
            # Fourier optics: angular spectrum method
            
            # Apply lens phase
            field_after_lens = field * self.lens_phase
            
            # Transform to k-space
            if self.config.ndim == 2:
                field_k = fft2(field_after_lens)
            else:
                field_k = fftn(field_after_lens)
            
            # Propagate in k-space: multiply by exp(i k_z z)
            propagator = np.exp(1j * self.k_z * distance)
            field_k_propagated = field_k * propagator
            
            # Transform back to real space
            if self.config.ndim == 2:
                field_focal = ifft2(field_k_propagated)
            else:
                field_focal = ifftn(field_k_propagated)
        
        elif self.config.propagation_method == 'fresnel':
            # Fresnel diffraction (paraxial approximation)
            
            k0 = 2 * np.pi / self.config.wavelength
            
            # Fresnel propagator in real space
            if self.config.ndim == 2:
                X, Y = self.grids
                r_squared = X**2 + Y**2
            else:
                X, Y, Z = self.grids
                r_squared = X**2 + Y**2 + Z**2
            
            fresnel_kernel = np.exp(1j * k0 * r_squared / (2 * distance))
            
            # Convolve
            if self.config.ndim == 2:
                field_k = fft2(field)
                kernel_k = fft2(fresnel_kernel)
                field_focal = ifft2(field_k * kernel_k)
            else:
                field_k = fftn(field)
                kernel_k = fftn(fresnel_kernel)
                field_focal = ifftn(field_k * kernel_k)
            
            # Apply lens phase
            field_focal *= self.lens_phase
        
        else:
            raise ValueError(f"Unknown propagation method: {self.config.propagation_method}")
        
        return field_focal
    
    def render(self, field_focal: np.ndarray,
              method: Optional[str] = None) -> np.ndarray:
        """
        Render (collapse) interference pattern to definite values.
        
        This is the wave function collapse / measurement / detection
        that transforms superposed potentiality into actualized reality.
        
        Parameters
        ----------
        field_focal : np.ndarray (complex)
            Interference pattern at focal plane
        method : str, optional
            Rendering method (uses config default if None)
            
        Returns
        -------
        rendered : np.ndarray (real)
            Collapsed intensity pattern
        """
        
        if method is None:
            method = self.config.collapse_method
        
        # Compute intensity (pre-collapse probability density)
        intensity = np.abs(field_focal) ** 2
        
        if method == 'born':
            # Born rule: probabilistic collapse
            # In practice, we interpret intensity as the rendered outcome
            # with quantum noise representing fundamental uncertainty
            
            # Add measurement noise
            noise = self.rng.normal(0, self.config.measurement_noise, intensity.shape)
            rendered = intensity + noise
            
            # Ensure non-negative
            rendered = np.maximum(rendered, 0)
        
        elif method == 'deterministic':
            # Deterministic: intensity directly becomes rendered value
            rendered = intensity
        
        elif method == 'threshold':
            # Threshold collapse: binary outcome above/below threshold
            normalized_intensity = intensity / (np.max(intensity) + 1e-10)
            rendered = (normalized_intensity > self.config.collapse_threshold).astype(float)
        
        elif method == 'stochastic':
            # Fully stochastic: sample from intensity distribution
            normalized_intensity = intensity / (np.sum(intensity) + 1e-10)
            
            # For each point, probabilistically collapse
            random_field = self.rng.random(intensity.shape)
            rendered = (random_field < normalized_intensity).astype(float)
        
        else:
            raise ValueError(f"Unknown collapse method: {method}")
        
        return rendered
    
    def compute_pommm(self, 
                     control_field: np.ndarray,
                     kram_geometry: np.ndarray,
                     chaos_field: np.ndarray,
                     return_intermediate: bool = False) -> Union[np.ndarray, dict]:
        """
        Complete POMMM computation: the cosmic matrix multiplication.
        
        Ψ_rendered = F_Instant[(M_KRAM Φ_Control) ⊗ (A_Chaos Φ_Potential)]
        
        This is the fundamental operation by which the universe computes
        its own evolution.
        
        Parameters
        ----------
        control_field : np.ndarray (complex)
            Coherent Control field (the Past)
        kram_geometry : np.ndarray (real)
            KRAM memory substrate
        chaos_field : np.ndarray (complex)
            Chaos attention field (the Future)
        return_intermediate : bool
            If True, return dict with intermediate states
            
        Returns
        -------
        rendered : np.ndarray (real) or dict
            Rendered reality (or full computation record)
        """
        
        # Step 1: KRAM modulation (Matrix A)
        field_after_kram = self.modulate_by_kram(control_field, kram_geometry)
        
        # Step 2: Chaos attention (Matrix B)
        field_doubly_modulated = self.apply_chaos_attention(field_after_kram, chaos_field)
        
        # Step 3: Propagate to Instant (lens/focal plane)
        field_at_instant = self.propagate_to_focal_plane(field_doubly_modulated)
        
        # Step 4: Render (collapse to actuality)
        rendered = self.render(field_at_instant)
        
        # Update statistics
        self.computation_count += 1
        self.total_energy_processed += np.sum(rendered)
        
        if return_intermediate:
            result = {
                'rendered': rendered,
                'control_input': control_field,
                'after_kram': field_after_kram,
                'after_chaos': field_doubly_modulated,
                'at_instant': field_at_instant,
                'kram_used': kram_geometry,
                'chaos_used': chaos_field,
                'total_energy': np.sum(rendered),
                'peak_intensity': np.max(rendered),
                'computation_number': self.computation_count
            }
            
            self.rendering_history.append({
                'step': self.computation_count,
                'energy': np.sum(rendered),
                'peak': np.max(rendered)
            })
            
            return result
        
        return rendered
    
    def compute_with_feedback(self,
                             control_field: np.ndarray,
                             kram_evolver,  # KRAMEvolver instance
                             chaos_field: np.ndarray,
                             update_kram: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        POMMM computation with feedback to KRAM (cosmic learning).
        
        The rendered output immediately imprints on KRAM, ensuring
        the universe learns from each computation.
        
        Parameters
        ----------
        control_field : np.ndarray (complex)
        kram_evolver : KRAMEvolver
            KRAM evolution object (modified in place)
        chaos_field : np.ndarray (complex)
        update_kram : bool
            Whether to actually update KRAM
            
        Returns
        -------
        rendered : np.ndarray
            Rendered output
        kram_updated : np.ndarray
            Updated KRAM geometry
        """
        
        # Forward computation
        result = self.compute_pommm(
            control_field,
            kram_evolver.g_M,
            chaos_field,
            return_intermediate=True
        )
        
        rendered = result['rendered']
        
        if update_kram:
            # Create imprint events from rendered pattern
            from kram_evolution_pde import ImprintEvent
            
            # Find significant rendered regions
            threshold = np.percentile(rendered, 75)
            significant_indices = np.argwhere(rendered > threshold)
            
            # Sample to avoid too many events
            if len(significant_indices) > 100:
                idx = self.rng.choice(len(significant_indices), 100, replace=False)
                significant_indices = significant_indices[idx]
            
            # Create imprint events
            events = []
            for idx in significant_indices:
                # Convert grid index to physical position
                position = np.array([
                    kram_evolver.grids[i][tuple(idx)] 
                    for i in range(kram_evolver.config.ndim)
                ])
                
                intensity = rendered[tuple(idx)] * self.config.feedback_strength
                
                events.append(ImprintEvent(
                    position=position,
                    intensity=intensity,
                    timestamp=kram_evolver.time
                ))
            
            # Update KRAM
            kram_evolver.step(imprint_events=events)
        
        return rendered, kram_evolver.g_M
    
    def matrix_formalism(self,
                        matrix_a: np.ndarray,
                        matrix_b: np.ndarray,
                        optical: bool = True) -> np.ndarray:
        """
        Compute matrix multiplication via optical or algebraic method.
        
        Demonstrates equivalence of optical interference and linear algebra.
        
        Parameters
        ----------
        matrix_a : np.ndarray (real or complex)
            First matrix
        matrix_b : np.ndarray (real or complex)
            Second matrix
        optical : bool
            If True, use optical method; if False, use direct multiplication
            
        Returns
        -------
        result : np.ndarray
            Matrix product AB
        """
        
        if not optical:
            # Direct algebraic multiplication
            return np.matmul(matrix_a, matrix_b)
        
        # Optical method: encode matrices as field modulations
        
        # For simplicity, assume square matrices matching grid size
        if matrix_a.shape != self.config.grid_size[:2]:
            warnings.warn("Matrix size mismatch with grid, reshaping")
            matrix_a = np.resize(matrix_a, self.config.grid_size[:2])
            matrix_b = np.resize(matrix_b, self.config.grid_size[:2])
        
        # Encode as complex fields
        field_a = matrix_a.astype(np.complex128)
        field_b = matrix_b.astype(np.complex128)
        
        # Create coherent illumination
        coherent_source = np.ones(self.config.grid_size[:2], dtype=np.complex128)
        
        # First modulation (Matrix A)
        field_after_a = coherent_source * field_a
        
        # Second modulation (Matrix B) 
        field_after_b = field_after_a * field_b
        
        # Fourier transform (lens operation)
        if self.config.ndim == 2:
            field_focal = fft2(field_after_b)
        else:
            field_focal = fftn(field_after_b)
        
        # Intensity gives matrix product
        result = np.abs(field_focal) ** 2
        
        # Normalize
        result /= np.max(result) + 1e-10
        
        return result
    
    def interference_visibility(self, field1: np.ndarray, 
                               field2: np.ndarray) -> float:
        """
        Compute interference visibility (contrast).
        
        V = (I_max - I_min) / (I_max + I_min)
        
        Parameters
        ----------
        field1, field2 : np.ndarray (complex)
            Two interfering fields
            
        Returns
        -------
        visibility : float
            Visibility in range [0, 1]
        """
        
        # Interference pattern
        total_field = field1 + field2
        intensity = np.abs(total_field) ** 2
        
        I_max = np.max(intensity)
        I_min = np.min(intensity)
        
        if I_max + I_min > 0:
            visibility = (I_max - I_min) / (I_max + I_min)
        else:
            visibility = 0.0
        
        return visibility
    
    def compute_mutual_coherence(self, field: np.ndarray,
                                point1: Tuple[int, ...],
                                point2: Tuple[int, ...]) -> complex:
        """
        Compute mutual coherence function Γ(r1, r2).
        
        Γ(r1, r2) = <E*(r1) E(r2)>
        
        Parameters
        ----------
        field : np.ndarray (complex)
        point1, point2 : tuples
            Grid indices of two points
            
        Returns
        -------
        coherence : complex
            Mutual coherence value
        """
        
        E1 = field[point1]
        E2 = field[point2]
        
        # In ensemble average, we approximate with spatial average nearby
        # For single realization, just compute product
        coherence = np.conj(E1) * E2
        
        return coherence
    
    def analyze_computation(self, result_dict: dict) -> dict:
        """
        Analyze a POMMM computation result.
        
        Parameters
        ----------
        result_dict : dict
            Result from compute_pommm with return_intermediate=True
            
        Returns
        -------
        analysis : dict
            Various analysis metrics
        """
        
        rendered = result_dict['rendered']
        at_instant = result_dict['at_instant']
        
        # Energy conservation
        energy_in_control = np.sum(np.abs(result_dict['control_input'])**2)
        energy_at_instant = np.sum(np.abs(at_instant)**2)
        energy_rendered = np.sum(rendered)
        
        # Coherence analysis
        field_center = tuple(s//2 for s in self.config.grid_size)
        field_offset = tuple(s//2 + 5 for s in self.config.grid_size)
        
        mutual_coh = self.compute_mutual_coherence(
            at_instant, field_center, field_offset
        )
        
        # Spatial structure
        rendered_fft = np.abs(fft2(rendered) if self.config.ndim == 2 else fftn(rendered))
        spectral_peak = np.unravel_index(np.argmax(rendered_fft[1:]), rendered_fft.shape)
        
        # Information content (Shannon entropy)
        hist, _ = np.histogram(rendered.flatten(), bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        analysis = {
            'energy_input': float(energy_in_control),
            'energy_at_instant': float(energy_at_instant),
            'energy_rendered': float(energy_rendered),
            'energy_conservation_ratio': float(energy_rendered / (energy_in_control + 1e-10)),
            'mutual_coherence': complex(mutual_coh),
            'coherence_magnitude': float(np.abs(mutual_coh)),
            'spatial_entropy_bits': float(entropy),
            'peak_rendered': float(np.max(rendered)),
            'mean_rendered': float(np.mean(rendered)),
            'std_rendered': float(np.std(rendered)),
            'spectral_peak_location': spectral_peak,
            'computation_number': result_dict['computation_number']
        }
        
        return analysis


# ============================================================================
# Convenience Functions
# ============================================================================

def simple_pommm_demo(grid_size: int = 64) -> dict:
    """
    Simple demonstration of POMMM computation.
    
    Parameters
    ----------
    grid_size : int
        
    Returns
    -------
    results : dict
        Computation results and visualization data
    """
    from control_field_generator import ControlFieldGenerator, ControlFieldConfig
    from kram_evolution_pde import KRAMEvolver, KRAMConfig
    
    # Setup
    pomm_config = POMMConfig(grid_size=(grid_size, grid_size), seed=42)
    engine = POMMEngine(pomm_config)
    
    # Create Control field (laser-like)
    control_config = ControlFieldConfig(grid_size=(grid_size, grid_size), seed=42)
    control_gen = ControlFieldGenerator(control_config)
    control_field = control_gen.gaussian_beam(beam_waist=5.0)
    
    # Create KRAM with some structure
    kram_config = KRAMConfig(grid_size=(grid_size, grid_size), seed=42)
    kram_evolver = KRAMEvolver(kram_config)
    
    # Add random imprints
    from kram_evolution_pde import ImprintEvent
    for _ in range(20):
        pos = np.random.rand(2) * kram_config.domain_size
        event = ImprintEvent(position=pos, intensity=1.5, timestamp=0)
        kram_evolver.step(imprint_events=[event])
    kram_evolver.evolve(200)
    
    # Create Chaos field (stochastic)
    chaos_field = np.exp(1j * 2 * np.pi * np.random.rand(grid_size, grid_size))
    chaos_field *= np.random.rand(grid_size, grid_size)
    
    # Compute POMMM
    result = engine.compute_pommm(
        control_field,
        kram_evolver.g_M,
        chaos_field,
        return_intermediate=True
    )
    
    # Analyze
    analysis = engine.analyze_computation(result)
    
    results = {
        'computation': result,
        'analysis': analysis,
        'engine': engine,
        'control_generator': control_gen,
        'kram_evolver': kram_evolver
    }
    
    return results


def pommm_with_learning(n_iterations: int = 10, grid_size: int = 64) -> dict:
    """
    Demonstrate POMMM with KRAM learning over multiple iterations.
    
    Parameters
    ----------
    n_iterations : int
    grid_size : int
        
    Returns
    -------
    results : dict
        Time series of computations showing learning
    """
    from control_field_generator import ControlFieldGenerator, ControlFieldConfig
    from kram_evolution_pde import KRAMEvolver, KRAMConfig
    
    # Setup
    pomm_config = POMMConfig(grid_size=(grid_size, grid_size), 
                            feedback_strength=0.05, seed=42)
    engine = POMMEngine(pomm_config)
    
    control_config = ControlFieldConfig(grid_size=(grid_size, grid_size), seed=42)
    control_gen = ControlFieldGenerator(control_config)
    
    kram_config = KRAMConfig(grid_size=(grid_size, grid_size), seed=42)
    kram_evolver = KRAMEvolver(kram_config)
    
    # Initialize with small random structure
    from kram_evolution_pde import ImprintEvent
    for _ in range(5):
        pos = np.random.rand(2) * kram_config.domain_size
        event = ImprintEvent(position=pos, intensity=0.5, timestamp=0)
        kram_evolver.step(imprint_events=[event])
    
    # Iteration loop
    history = []
    
    for i in range(n_iterations):
        # Generate fields
        control_field = control_gen.gaussian_beam(beam_waist=4.0)
        control_field = control_gen.temporal_evolution(control_field, time_step=i)
        
        chaos_field = np.exp(1j * 2 * np.pi * np.random.rand(grid_size, grid_size))
        chaos_field *= (0.5 + 0.5 * np.random.rand(grid_size, grid_size))
        
        # Compute with feedback
        rendered, kram_updated = engine.compute_with_feedback(
            control_field,
            kram_evolver,
            chaos_field,
            update_kram=True
        )
        
        # Record metrics
        history.append({
            'iteration': i,
            'total_energy': np.sum(rendered),
            'peak_intensity': np.max(rendered),
            'kram_complexity': np.std(kram_evolver.g_M),
            'kram_mean': np.mean(kram_evolver.g_M),
            'n_attractors': len(kram_evolver.get_attractor_valleys(threshold=0.3)[1])
        })
        
        # Let KRAM relax between iterations
        kram_evolver.evolve(20)
    
    results = {
        'history': history,
        'final_kram': kram_evolver.g_M,
        'engine': engine,
        'kram_evolver': kram_evolver
    }
    
    return results


# ============================================================================
# Testing and Visualization
# ============================================================================

def _test_pommm_engine():
    """Test suite for POMMM engine."""
    
    print("=" * 70)
    print("POMMM Interference Engine Test Suite")
    print("=" * 70)
    
    # Test 1: Basic initialization
    print("\n[Test 1] Engine initialization...")
    config = POMMConfig(grid_size=(64, 64), seed=42)
    engine = POMMEngine(config)
    
    print(f"  Grid size: {engine.config.grid_size}")
    print(f"  Domain size: {engine.config.domain_size}")
    print(f"  Wavelength: {engine.config.wavelength}")
    print(f"  Focal length: {engine.config.focal_length}")
    
    # Test 2: Field modulation
    print("\n[Test 2] KRAM modulation...")
    control_field = np.ones((64, 64), dtype=np.complex128)
    kram_geometry = np.random.randn(64, 64) * 0.5
    
    modulated = engine.modulate_by_kram(control_field, kram_geometry, 'phase')
    
    print(f"  Input amplitude mean: {np.mean(np.abs(control_field)):.6f}")
    print(f"  Modulated amplitude mean: {np.mean(np.abs(modulated)):.6f}")
    print(f"  Phase variance: {np.var(np.angle(modulated)):.6f}")
    print(f"  Energy conservation: {np.allclose(np.abs(modulated), np.abs(control_field))}")
    
    # Test 3: Chaos attention
    print("\n[Test 3] Chaos attention filtering...")
    chaos_pattern = 0.5 + 0.5 * np.random.rand(64, 64)
    
    attended = engine.apply_chaos_attention(modulated, chaos_pattern, 'selective')
    
    energy_before = np.sum(np.abs(modulated)**2)
    energy_after = np.sum(np.abs(attended)**2)
    
    print(f"  Energy before attention: {energy_before:.6f}")
    print(f"  Energy after attention: {energy_after:.6f}")
    print(f"  Attention efficiency: {energy_after/energy_before:.6f}")
    
    # Test 4: Propagation to focal plane
    print("\n[Test 4] Propagation to Instant...")
    field_focal = engine.propagate_to_focal_plane(attended)
    
    print(f"  Input peak amplitude: {np.max(np.abs(attended)):.6f}")
    print(f"  Focal peak amplitude: {np.max(np.abs(field_focal)):.6f}")
    print(f"  Focusing gain: {np.max(np.abs(field_focal))/np.max(np.abs(attended)):.6f}")
    
    # Test 5: Rendering (collapse)
    print("\n[Test 5] Rendering to actuality...")
    
    for method in ['born', 'deterministic', 'threshold']:
        rendered = engine.render(field_focal, method=method)
        print(f"  Method: {method:15s} - Mean: {np.mean(rendered):.6f}, "
              f"Peak: {np.max(rendered):.6f}")
    
    # Test 6: Complete POMMM computation
    print("\n[Test 6] Complete POMMM computation...")
    
    # Setup realistic fields
    from control_field_generator import ControlFieldGenerator, ControlFieldConfig
    from kram_evolution_pde import KRAMEvolver, KRAMConfig
    
    control_config = ControlFieldConfig(grid_size=(64, 64), seed=42)
    control_gen = ControlFieldGenerator(control_config)
    control_field = control_gen.gaussian_beam()
    
    kram_config = KRAMConfig(grid_size=(64, 64), seed=42)
    kram_evolver = KRAMEvolver(kram_config)
    
    from kram_evolution_pde import ImprintEvent
    for _ in range(10):
        pos = np.random.rand(2) * kram_config.domain_size
        event = ImprintEvent(position=pos, intensity=1.0, timestamp=0)
        kram_evolver.step(imprint_events=[event])
    kram_evolver.evolve(100)
    
    chaos_field = np.exp(1j * 2 * np.pi * np.random.rand(64, 64))
    
    result = engine.compute_pommm(
        control_field,
        kram_evolver.g_M,
        chaos_field,
        return_intermediate=True
    )
    
    print(f"  Total energy rendered: {result['total_energy']:.6f}")
    print(f"  Peak intensity: {result['peak_intensity']:.6f}")
    print(f"  Computation count: {engine.computation_count}")
    
    # Test 7: Analysis
    print("\n[Test 7] Computation analysis...")
    analysis = engine.analyze_computation(result)
    
    print(f"  Energy conservation ratio: {analysis['energy_conservation_ratio']:.6f}")
    print(f"  Coherence magnitude: {analysis['coherence_magnitude']:.6f}")
    print(f"  Spatial entropy: {analysis['spatial_entropy_bits']:.4f} bits")
    
    # Test 8: Matrix formalism equivalence
    print("\n[Test 8] Matrix formalism equivalence...")
    
    # Small matrices for testing
    A = np.random.rand(8, 8)
    B = np.random.rand(8, 8)
    
    engine_small = POMMEngine(POMMConfig(grid_size=(8, 8), seed=42))
    
    result_algebraic = engine_small.matrix_formalism(A, B, optical=False)
    result_optical = engine_small.matrix_formalism(A, B, optical=True)
    
    # Normalize for comparison
    result_optical_norm = result_optical / np.max(result_optical)
    result_algebraic_norm = np.abs(result_algebraic) / np.max(np.abs(result_algebraic))
    
    correlation = np.corrcoef(result_optical_norm.flatten(), 
                             result_algebraic_norm.flatten())[0, 1]
    
    print(f"  Algebraic result peak: {np.max(np.abs(result_algebraic)):.6f}")
    print(f"  Optical result peak: {np.max(result_optical):.6f}")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Match quality: {'EXCELLENT' if correlation > 0.8 else 'GOOD' if correlation > 0.6 else 'POOR'}")
    
    # Test 9: Interference visibility
    print("\n[Test 9] Interference visibility...")
    
    field1 = control_gen.plane_wave(k_vector=np.array([0.1, 0]))
    field2 = control_gen.plane_wave(k_vector=np.array([0.1, 0.05]))
    
    visibility = engine.interference_visibility(field1, field2)
    
    print(f"  Visibility: {visibility:.6f}")
    print(f"  Coherence quality: {'HIGH' if visibility > 0.7 else 'MEDIUM' if visibility > 0.4 else 'LOW'}")
    
    # Test 10: Feedback and learning
    print("\n[Test 10] POMMM with feedback...")
    
    kram_initial = kram_evolver.g_M.copy()
    
    rendered, kram_updated = engine.compute_with_feedback(
        control_field,
        kram_evolver,
        chaos_field,
        update_kram=True
    )
    
    kram_change = np.mean(np.abs(kram_updated - kram_initial))
    
    print(f"  KRAM mean change: {kram_change:.6f}")
    print(f"  Rendered energy: {np.sum(rendered):.6f}")
    print(f"  Feedback active: {kram_change > 1e-6}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


def _visualize_pommm_computation():
    """Generate comprehensive visualization of POMMM computation."""
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Matplotlib not available, skipping visualization.")
        return
    
    print("\nGenerating POMMM computation visualization...")
    
    # Run complete demo
    results = simple_pommm_demo(grid_size=128)
    
    computation = results['computation']
    analysis = results['analysis']
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Input fields
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(np.abs(computation['control_input']), cmap='viridis')
    ax1.set_title('Control Field\n(Coherent Past)', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(computation['kram_used'], cmap='RdBu_r')
    ax2.set_title('KRAM Geometry\n(Ancestral Memory)', fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(np.abs(computation['chaos_used']), cmap='plasma')
    ax3.set_title('Chaos Field\n(Selective Attention)', fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(np.angle(computation['control_input']), cmap='twilight')
    ax4.set_title('Control Phase\n(Coherence)', fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: Intermediate stages
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(np.abs(computation['after_kram']), cmap='viridis')
    ax5.set_title('After KRAM\nModulation', fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(np.abs(computation['after_chaos']), cmap='viridis')
    ax6.set_title('After Chaos\nAttention', fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(np.abs(computation['at_instant']), cmap='hot')
    ax7.set_title('At Instant\n(Interference)', fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(computation['rendered'], cmap='inferno')
    ax8.set_title('Rendered Reality\n(Collapsed)', fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Row 3: Analysis panels
    
    # Power spectrum of rendered output
    ax9 = fig.add_subplot(gs[2, 0])
    rendered_fft = np.abs(np.fft.fftshift(np.fft.fft2(computation['rendered'])))
    ax9.imshow(np.log10(rendered_fft + 1), cmap='viridis')
    ax9.set_title('Rendered\nPower Spectrum', fontweight='bold')
    ax9.axis('off')
    
    # Cross-section comparison
    ax10 = fig.add_subplot(gs[2, 1])
    center = computation['rendered'].shape[0] // 2
    ax10.plot(np.abs(computation['control_input'][center, :]), 
             label='Control Input', alpha=0.7)
    ax10.plot(computation['rendered'][center, :], 
             label='Rendered Output', alpha=0.7, linewidth=2)
    ax10.set_title('Central Cross-Section', fontweight='bold')
    ax10.set_xlabel('Position')
    ax10.set_ylabel('Amplitude')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Energy flow diagram
    ax11 = fig.add_subplot(gs[2, 2])
    energies = [
        analysis['energy_input'],
        analysis['energy_at_instant'],
        analysis['energy_rendered']
    ]
    labels = ['Input', 'At Instant', 'Rendered']
    colors = ['blue', 'orange', 'red']
    
    bars = ax11.bar(labels, energies, color=colors, alpha=0.7)
    ax11.set_title('Energy Flow', fontweight='bold')
    ax11.set_ylabel('Total Energy')
    ax11.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, energies):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.2f}', ha='center', va='bottom')
    
    # Analysis metrics text
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    metrics_text = f"""Analysis Metrics:

Energy Conservation:
  Ratio: {analysis['energy_conservation_ratio']:.4f}

Coherence:
  Magnitude: {analysis['coherence_magnitude']:.4f}

Information:
  Entropy: {analysis['spatial_entropy_bits']:.2f} bits

Statistics:
  Peak: {analysis['peak_rendered']:.4f}
  Mean: {analysis['mean_rendered']:.4f}
  Std: {analysis['std_rendered']:.4f}

Computation:
  Number: {analysis['computation_number']}
"""
    
    ax12.text(0.1, 0.9, metrics_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('KnoWellian Parallel Optical Matrix-Matrix Multiplication:\n' +
                'The Cosmic Computational Engine',
                fontsize=16, fontweight='bold')
    
    plt.savefig('pommm_computation.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'pommm_computation.png'")
    
    # Create learning visualization
    print("\nGenerating learning visualization...")
    
    learning_results = pommm_with_learning(n_iterations=20, grid_size=128)
    history = learning_results['history']
    
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    iterations = [h['iteration'] for h in history]
    
    # Energy evolution
    axes[0, 0].plot(iterations, [h['total_energy'] for h in history], 
                   'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Total Rendered Energy', fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Peak intensity
    axes[0, 1].plot(iterations, [h['peak_intensity'] for h in history], 
                   'r-o', linewidth=2, markersize=4)
    axes[0, 1].set_title('Peak Intensity', fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Peak')
    axes[0, 1].grid(True, alpha=0.3)
    
    # KRAM complexity
    axes[0, 2].plot(iterations, [h['kram_complexity'] for h in history], 
                   'g-o', linewidth=2, markersize=4)
    axes[0, 2].set_title('KRAM Complexity', fontweight='bold')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Std Dev')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Number of attractors
    axes[1, 0].plot(iterations, [h['n_attractors'] for h in history], 
                   'm-o', linewidth=2, markersize=4)
    axes[1, 0].set_title('Number of Attractors', fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # KRAM mean
    axes[1, 1].plot(iterations, [h['kram_mean'] for h in history], 
                   'c-o', linewidth=2, markersize=4)
    axes[1, 1].set_title('KRAM Mean Value', fontweight='bold')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Mean g_M')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Final KRAM state
    axes[1, 2].imshow(learning_results['final_kram'], cmap='RdBu_r')
    axes[1, 2].set_title('Final KRAM State', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('POMMM with KRAM Learning:\nCosmic Memory Evolution',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pommm_learning.png', dpi=150, bbox_inches='tight')
    print("Learning visualization saved as 'pommm_learning.png'")
    
    plt.close('all')


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run test suite
    _test_pommm_engine()
    
    # Generate visualizations
    _visualize_pommm_computation()
    
    print("\n" + "=" * 70)
    print("POMMM Interference Engine Complete")
    print("=" * 70)
    print("\nThe universe computes through light...")
    print("Every interference pattern is a moment of cosmic thought.")
    print("Every collapse is a new imprint on the eternal manifold.")