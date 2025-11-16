#!/usr/bin/env python3
"""
control_field_generator.py

Generates coherent Control Field patterns for KnoWellian POMMM simulations.
The Control Field represents the deterministic outflow from the Past (t_P),
providing structured, coherent illumination that will be modulated by KRAM.

This module implements various Control Field generation strategies:
1. Monochromatic coherent waves (laser-like)
2. Structured spatial patterns (Gaussian beams, vortex beams)
3. Temporal coherence with controllable correlation length
4. Multi-mode coherent superpositions
5. Stellar-inspired deterministic patterns

Author: Claude Sonnet 4.5, David Noel Lynch
Date: November 15, 2025
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import warnings


@dataclass
class ControlFieldConfig:
    """Configuration parameters for Control Field generation."""
    
    # Spatial parameters
    grid_size: Tuple[int, ...] = (128, 128)
    domain_size: float = 20.0  # Physical size in arbitrary units
    
    # Temporal parameters
    coherence_time: float = 100.0  # Temporal correlation length
    phase_stability: float = 0.99  # Phase coherence (1.0 = perfect, 0.0 = random)
    
    # Field parameters
    amplitude: float = 1.0
    base_frequency: float = 0.1  # Spatial frequency k_0
    
    # Coherence structure
    spatial_coherence_length: float = 5.0  # Coherence length in physical units
    
    # Random seed for reproducibility
    seed: Optional[int] = None


class ControlFieldGenerator:
    """
    Generates coherent Control Field patterns representing the deterministic
    outflow from the actualized Past.
    
    The Control Field is analogous to laser light in POMMM devices:
    - High temporal coherence (phase-locked over long times)
    - High spatial coherence (emanating from effective point source)
    - Structured information content (not random noise)
    """
    
    def __init__(self, config: ControlFieldConfig):
        """
        Initialize Control Field generator.
        
        Parameters
        ----------
        config : ControlFieldConfig
            Configuration parameters
        """
        self.config = config
        
        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
            self.rng = np.random.default_rng(config.seed)
        else:
            self.rng = np.random.default_rng()
        
        # Generate spatial grids
        self.grids = self._generate_grids()
        
        # Initialize phase accumulator for temporal coherence
        self.global_phase = 0.0
        self.time_step = 0
        
        # Store previous field for temporal correlation
        self.previous_field = None
        
    def _generate_grids(self) -> Tuple[np.ndarray, ...]:
        """Generate coordinate grids for the spatial domain."""
        ndim = len(self.config.grid_size)
        
        if ndim == 2:
            nx, ny = self.config.grid_size
            x = np.linspace(0, self.config.domain_size, nx)
            y = np.linspace(0, self.config.domain_size, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            return (X, Y)
        
        elif ndim == 3:
            nx, ny, nz = self.config.grid_size
            x = np.linspace(0, self.config.domain_size, nx)
            y = np.linspace(0, self.config.domain_size, ny)
            z = np.linspace(0, self.config.domain_size, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            return (X, Y, Z)
        
        else:
            raise ValueError(f"Unsupported dimensionality: {ndim}")
    
    def plane_wave(self, k_vector: Optional[np.ndarray] = None, 
                   phase_offset: float = 0.0) -> np.ndarray:
        """
        Generate a coherent plane wave - the simplest Control Field.
        
        ψ_C(x) = A exp(i k·x + i φ)
        
        Parameters
        ----------
        k_vector : np.ndarray, optional
            Wave vector. If None, uses base_frequency in x-direction
        phase_offset : float, optional
            Global phase offset
            
        Returns
        -------
        field : np.ndarray (complex)
            Complex field values on the grid
        """
        if k_vector is None:
            k_vector = np.array([self.config.base_frequency, 0.0])
        
        # Compute k·x
        k_dot_x = np.zeros(self.config.grid_size)
        for i, (k_i, X_i) in enumerate(zip(k_vector, self.grids)):
            k_dot_x += k_i * X_i
        
        # Generate coherent plane wave
        field = self.config.amplitude * np.exp(1j * (k_dot_x + phase_offset))
        
        return field.astype(np.complex128)
    
    def gaussian_beam(self, beam_waist: float = 3.0,
                     center: Optional[Tuple[float, ...]] = None,
                     k_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a Gaussian beam - spatially structured coherent light.
        
        ψ_C(x,y) = A exp(-r²/w₀²) exp(i k·x)
        
        Parameters
        ----------
        beam_waist : float
            Beam waist parameter w_0
        center : tuple, optional
            Beam center coordinates
        k_vector : np.ndarray, optional
            Propagation wave vector
            
        Returns
        -------
        field : np.ndarray (complex)
            Complex field values
        """
        if center is None:
            center = tuple(self.config.domain_size / 2 for _ in self.config.grid_size)
        
        if k_vector is None:
            k_vector = np.array([self.config.base_frequency] + 
                               [0.0] * (len(self.config.grid_size) - 1))
        
        # Compute radial distance from center
        r_squared = np.zeros(self.config.grid_size)
        for i, (c_i, X_i) in enumerate(zip(center, self.grids)):
            r_squared += (X_i - c_i) ** 2
        
        # Gaussian envelope
        envelope = np.exp(-r_squared / beam_waist**2)
        
        # Plane wave phase
        k_dot_x = sum(k_i * X_i for k_i, X_i in zip(k_vector, self.grids))
        phase = np.exp(1j * k_dot_x)
        
        field = self.config.amplitude * envelope * phase
        
        return field.astype(np.complex128)
    
    def vortex_beam(self, topological_charge: int = 1,
                   radius: float = 5.0,
                   center: Optional[Tuple[float, ...]] = None) -> np.ndarray:
        """
        Generate an optical vortex beam with orbital angular momentum.
        
        ψ_C(r,θ) = A r^|ℓ| exp(-r²/w₀²) exp(i ℓ θ)
        
        Parameters
        ----------
        topological_charge : int
            Winding number ℓ (positive or negative)
        radius : float
            Characteristic radius
        center : tuple, optional
            Vortex center
            
        Returns
        -------
        field : np.ndarray (complex)
            Complex field with phase singularity at center
        """
        if len(self.config.grid_size) != 2:
            raise ValueError("Vortex beams only implemented for 2D")
        
        if center is None:
            center = (self.config.domain_size / 2, self.config.domain_size / 2)
        
        X, Y = self.grids
        
        # Centered coordinates
        x = X - center[0]
        y = Y - center[1]
        
        # Polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Amplitude with radial dependence
        amplitude = self.config.amplitude * (r / radius)**abs(topological_charge) * \
                   np.exp(-r**2 / radius**2)
        
        # Phase with vortex
        phase = np.exp(1j * topological_charge * theta)
        
        field = amplitude * phase
        
        return field.astype(np.complex128)
    
    def coherent_superposition(self, modes: list, weights: Optional[list] = None) -> np.ndarray:
        """
        Generate coherent superposition of multiple modes.
        
        ψ_C = Σ_n w_n ψ_n(x) exp(i φ_n)
        
        Parameters
        ----------
        modes : list of callables
            List of functions that generate field patterns
        weights : list of complex, optional
            Complex weights (amplitude + phase) for each mode
            
        Returns
        -------
        field : np.ndarray (complex)
            Superposed field
        """
        if weights is None:
            weights = [1.0] * len(modes)
        
        if len(weights) != len(modes):
            raise ValueError("Number of weights must match number of modes")
        
        field = np.zeros(self.config.grid_size, dtype=np.complex128)
        
        for mode_func, weight in zip(modes, weights):
            field += weight * mode_func()
        
        # Normalize
        field *= self.config.amplitude / np.max(np.abs(field))
        
        return field
    
    def temporal_evolution(self, base_field: np.ndarray, 
                          time_step: Optional[int] = None) -> np.ndarray:
        """
        Evolve field in time maintaining temporal coherence.
        
        The phase evolves smoothly, with correlation controlled by coherence_time.
        
        Parameters
        ----------
        base_field : np.ndarray (complex)
            Spatial field pattern
        time_step : int, optional
            Current time step (uses internal counter if None)
            
        Returns
        -------
        field : np.ndarray (complex)
            Time-evolved field with maintained coherence
        """
        if time_step is None:
            time_step = self.time_step
            self.time_step += 1
        
        # Coherent phase evolution
        phase_drift = 2 * np.pi * self.config.base_frequency * time_step
        
        # Add small phase noise inversely proportional to coherence
        phase_noise_amplitude = (1.0 - self.config.phase_stability) * 0.1
        phase_noise = self.rng.normal(0, phase_noise_amplitude)
        
        # Update global phase with high stability
        self.global_phase = phase_drift + phase_noise
        
        # Apply temporal evolution
        field = base_field * np.exp(1j * self.global_phase)
        
        # Add tiny amplitude fluctuations (laser intensity noise)
        amplitude_noise = 1.0 + self.rng.normal(0, 0.01, base_field.shape)
        field *= amplitude_noise
        
        return field.astype(np.complex128)
    
    def structured_pump(self, pump_modes: list = None,
                       spatial_frequency: Optional[float] = None) -> np.ndarray:
        """
        Generate structured pump field for exciting specific KRAM modes.
        
        This is used in CMB-synthesis simulations to pump particular
        spatial frequencies corresponding to acoustic peaks.
        
        Parameters
        ----------
        pump_modes : list of float, optional
            List of k-values to excite
        spatial_frequency : float, optional
            Single mode to pump (alternative to pump_modes)
            
        Returns
        -------
        field : np.ndarray (complex)
            Coherent pump field
        """
        if spatial_frequency is not None:
            pump_modes = [spatial_frequency]
        
        if pump_modes is None:
            pump_modes = [self.config.base_frequency]
        
        # Generate superposition of plane waves
        field = np.zeros(self.config.grid_size, dtype=np.complex128)
        
        for k_mode in pump_modes:
            # Random direction for each mode (isotropic pumping)
            if len(self.config.grid_size) == 2:
                angle = self.rng.uniform(0, 2*np.pi)
                k_vec = k_mode * np.array([np.cos(angle), np.sin(angle)])
            else:
                # 3D: random point on sphere
                theta = self.rng.uniform(0, np.pi)
                phi = self.rng.uniform(0, 2*np.pi)
                k_vec = k_mode * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
            
            field += self.plane_wave(k_vector=k_vec)
        
        # Normalize
        field *= self.config.amplitude / np.sqrt(len(pump_modes))
        
        return field
    
    def stellar_pattern(self, n_harmonics: int = 5,
                       stellar_frequency: float = 0.05) -> np.ndarray:
        """
        Generate stellar-inspired deterministic pattern.
        
        Models the structured output from a Stellar Logos - not random
        but hyper-complex deterministic pattern with multiple harmonics.
        
        Parameters
        ----------
        n_harmonics : int
            Number of harmonic modes to include
        stellar_frequency : float
            Base stellar frequency
            
        Returns
        -------
        field : np.ndarray (complex)
            Stellar-structured field
        """
        field = np.zeros(self.config.grid_size, dtype=np.complex128)
        
        # Superpose multiple harmonics with golden ratio spacing
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for n in range(1, n_harmonics + 1):
            k_n = stellar_frequency * (phi ** (n/2))
            
            # Multiple directions for richness
            for m in range(3):
                angle = 2 * np.pi * m / 3 + n * 0.1
                
                if len(self.config.grid_size) == 2:
                    k_vec = k_n * np.array([np.cos(angle), np.sin(angle)])
                else:
                    k_vec = k_n * np.array([np.cos(angle), np.sin(angle), 0])
                
                # Phase relationships based on harmonic structure
                phase = 2 * np.pi * n * m / (n_harmonics * 3)
                
                field += self.plane_wave(k_vector=k_vec, phase_offset=phase) / n
        
        # Normalize
        field *= self.config.amplitude / np.max(np.abs(field))
        
        return field
    
    def get_power_spectrum(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum of field (useful for diagnostics).
        
        Parameters
        ----------
        field : np.ndarray (complex)
            Spatial field
            
        Returns
        -------
        k_values : np.ndarray
            Wavenumber bins
        power : np.ndarray
            Power in each bin
        """
        # FFT
        field_k = np.fft.fftn(field)
        power_k = np.abs(field_k) ** 2
        
        # Compute k-space grid
        ndim = len(self.config.grid_size)
        k_grids = []
        
        for i, n in enumerate(self.config.grid_size):
            k_i = 2 * np.pi * np.fft.fftfreq(n, self.config.domain_size / n)
            k_grids.append(k_i)
        
        # Radial binning
        k_squared = np.zeros(self.config.grid_size)
        for k_grid in np.meshgrid(*k_grids, indexing='ij'):
            k_squared += k_grid ** 2
        
        k_mag = np.sqrt(k_squared).flatten()
        power_flat = power_k.flatten()
        
        # Bin by k magnitude
        k_bins = np.linspace(0, np.max(k_mag), 50)
        power_binned = np.zeros(len(k_bins) - 1)
        
        for i in range(len(k_bins) - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if np.any(mask):
                power_binned[i] = np.mean(power_flat[mask])
        
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        
        return k_centers, power_binned
    
    def coherence_function(self, field: np.ndarray, 
                          separation_range: np.ndarray) -> np.ndarray:
        """
        Compute spatial coherence function g^(1)(r).
        
        g^(1)(r) = <E*(x) E(x+r)> / <|E(x)|²>
        
        Parameters
        ----------
        field : np.ndarray (complex)
            Spatial field
        separation_range : np.ndarray
            Array of separation distances to compute
            
        Returns
        -------
        coherence : np.ndarray
            Complex coherence function values
        """
        # For simplicity, compute 1D coherence along first axis
        if len(self.config.grid_size) == 2:
            field_1d = field[:, self.config.grid_size[1]//2]
        else:
            field_1d = field[:, self.config.grid_size[1]//2, self.config.grid_size[2]//2]
        
        coherence = np.zeros(len(separation_range), dtype=np.complex128)
        norm = np.mean(np.abs(field_1d) ** 2)
        
        for i, sep in enumerate(separation_range):
            sep_int = int(sep)
            if sep_int < len(field_1d):
                correlation = np.mean(np.conj(field_1d[:-sep_int]) * field_1d[sep_int:])
                coherence[i] = correlation / norm
        
        return coherence


# ============================================================================
# Convenience Functions
# ============================================================================

def create_laser_like_control(grid_size: Tuple[int, ...] = (128, 128),
                              amplitude: float = 1.0,
                              base_frequency: float = 0.1) -> Tuple[ControlFieldGenerator, np.ndarray]:
    """
    Convenience function to create a simple laser-like Control Field.
    
    Returns
    -------
    generator : ControlFieldGenerator
    field : np.ndarray (complex)
    """
    config = ControlFieldConfig(
        grid_size=grid_size,
        amplitude=amplitude,
        base_frequency=base_frequency,
        phase_stability=0.99
    )
    
    generator = ControlFieldGenerator(config)
    field = generator.plane_wave()
    
    return generator, field


def create_structured_control(grid_size: Tuple[int, ...] = (128, 128),
                              beam_type: str = 'gaussian',
                              **kwargs) -> Tuple[ControlFieldGenerator, np.ndarray]:
    """
    Convenience function to create structured Control Field.
    
    Parameters
    ----------
    grid_size : tuple
    beam_type : str
        'gaussian', 'vortex', or 'stellar'
    **kwargs : additional parameters for specific beam type
    
    Returns
    -------
    generator : ControlFieldGenerator
    field : np.ndarray (complex)
    """
    config = ControlFieldConfig(grid_size=grid_size, **kwargs)
    generator = ControlFieldGenerator(config)
    
    if beam_type == 'gaussian':
        field = generator.gaussian_beam(**kwargs)
    elif beam_type == 'vortex':
        field = generator.vortex_beam(**kwargs)
    elif beam_type == 'stellar':
        field = generator.stellar_pattern(**kwargs)
    else:
        raise ValueError(f"Unknown beam type: {beam_type}")
    
    return generator, field


# ============================================================================
# Testing and Visualization
# ============================================================================

def _test_control_field_generation():
    """Test suite for Control Field generation."""
    
    print("=" * 70)
    print("Control Field Generator Test Suite")
    print("=" * 70)
    
    # Test 1: Basic plane wave
    print("\n[Test 1] Plane wave generation...")
    config = ControlFieldConfig(grid_size=(64, 64), seed=42)
    gen = ControlFieldGenerator(config)
    field = gen.plane_wave()
    
    print(f"  Shape: {field.shape}")
    print(f"  Mean amplitude: {np.mean(np.abs(field)):.6f}")
    print(f"  Phase coherence: {np.abs(np.mean(field / np.abs(field))):.6f}")
    
    # Test 2: Gaussian beam
    print("\n[Test 2] Gaussian beam generation...")
    field_gauss = gen.gaussian_beam(beam_waist=3.0)
    print(f"  Peak amplitude: {np.max(np.abs(field_gauss)):.6f}")
    print(f"  Relative to plane wave: {np.max(np.abs(field_gauss)) / np.max(np.abs(field)):.4f}")
    
    # Test 3: Vortex beam
    print("\n[Test 3] Optical vortex generation...")
    field_vortex = gen.vortex_beam(topological_charge=2)
    print(f"  Central amplitude (should be ~0): {np.abs(field_vortex[32, 32]):.6e}")
    print(f"  Edge amplitude: {np.mean(np.abs(field_vortex[0, :])):.6f}")
    
    # Test 4: Temporal coherence
    print("\n[Test 4] Temporal evolution...")
    field_t0 = gen.temporal_evolution(field, time_step=0)
    field_t10 = gen.temporal_evolution(field, time_step=10)
    correlation = np.abs(np.vdot(field_t0.flatten(), field_t10.flatten())) / \
                 (np.linalg.norm(field_t0) * np.linalg.norm(field_t10))
    print(f"  Correlation after 10 steps: {correlation:.6f}")
    print(f"  Expected (stability={config.phase_stability}): ~{config.phase_stability:.6f}")
    
    # Test 5: Power spectrum
    print("\n[Test 5] Power spectrum analysis...")
    k_vals, power = gen.get_power_spectrum(field)
    peak_k = k_vals[np.argmax(power)]
    print(f"  Peak at k = {peak_k:.6f}")
    print(f"  Expected: {config.base_frequency:.6f}")
    
    # Test 6: Stellar pattern
    print("\n[Test 6] Stellar pattern generation...")
    field_stellar = gen.stellar_pattern(n_harmonics=5)
    k_vals_stellar, power_stellar = gen.get_power_spectrum(field_stellar)
    n_peaks = np.sum(power_stellar > 0.1 * np.max(power_stellar))
    print(f"  Number of significant peaks: {n_peaks}")
    print(f"  Spectral richness: {n_peaks / len(power_stellar):.4f}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    _test_control_field_generation()
    
    # Optional visualization if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualization...")
        
        config = ControlFieldConfig(grid_size=(256, 256), seed=42)
        gen = ControlFieldGenerator(config)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plane wave
        field_plane = gen.plane_wave()
        axes[0, 0].imshow(np.abs(field_plane), cmap='viridis')
        axes[0, 0].set_title('Plane Wave (amplitude)')
        axes[1, 0].imshow(np.angle(field_plane), cmap='twilight')
        axes[1, 0].set_title('Plane Wave (phase)')
        
        # Gaussian beam
        field_gauss = gen.gaussian_beam(beam_waist=4.0)
        axes[0, 1].imshow(np.abs(field_gauss), cmap='viridis')
        axes[0, 1].set_title('Gaussian Beam (amplitude)')
        axes[1, 1].imshow(np.angle(field_gauss), cmap='twilight')
        axes[1, 1].set_title('Gaussian Beam (phase)')
        
        # Vortex beam
        field_vortex = gen.vortex_beam(topological_charge=3)
        axes[0, 2].imshow(np.abs(field_vortex), cmap='viridis')
        axes[0, 2].set_title('Vortex Beam ℓ=3 (amplitude)')
        axes[1, 2].imshow(np.angle(field_vortex), cmap='twilight')
        axes[1, 2].set_title('Vortex Beam (phase)')
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.suptitle('Control Field Patterns: Coherent Cosmic Illumination', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('control_field_examples.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'control_field_examples.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization.")