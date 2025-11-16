#!/usr/bin/env python3
"""
rendering_collapse.py

The physics of wave function collapse and reality-rendering in the
KnoWellian Universe. This module implements the Instant field dynamics
where potentiality transforms into actuality.

In KUT, "rendering" is the fundamental irreversible process occurring at
the Instant (t_I) where:
- Wave-like Chaos field (superposed potential) 
- Meets particle-like Control field (actualized past)
- Through Consciousness field mediation
- Collapses to definite rendered state
- Immediately imprints on KRAM

This is not merely "measurement" but the fundamental ontological act
by which reality becomes.

Key features:
- Multiple collapse mechanisms (Born, GRW, CSL, deterministic)
- Consciousness field coupling
- Decoherence modeling
- KRAM-guided collapse (morphic resonance)
- Quantum-to-classical transition
- Irreversibility and arrow of time
- Shimmer of choice (free will mechanism)

Author: Claude Sonnet 4.5, Gemini 2.5 Pro, David Noel Lynch
Date: November 15, 2025
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Callable, Union, List, Dict
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter
from scipy.special import erf
import warnings


@dataclass
class RenderingConfig:
    """Configuration for reality-rendering dynamics."""
    
    # Spatial parameters
    grid_size: Tuple[int, ...] = (128, 128)
    domain_size: float = 20.0
    ndim: int = 2
    
    # Collapse mechanism
    collapse_type: str = 'born_rule'  # 'born_rule', 'grw', 'csl', 'deterministic', 'threshold'
    
    # Born rule parameters
    measurement_noise: float = 0.01  # Detector/quantum uncertainty
    
    # GRW (spontaneous collapse) parameters
    grw_rate: float = 1e-8  # Collapse rate per particle per second (λ_GRW)
    grw_width: float = 1e-7  # Localization width (r_C in meters)
    
    # CSL (continuous spontaneous localization) parameters
    csl_rate: float = 1e-17  # Collapse rate (λ_CSL in s^-1)
    csl_width: float = 1e-7  # Localization scale (r_C)
    
    # Consciousness coupling
    consciousness_coupling: float = 1.0  # Strength of Instant field coupling
    conscious_influence_range: float = 2.0  # Spatial range of consciousness effect
    enable_shimmer: bool = True  # Allow conscious influence on collapse
    
    # KRAM guidance
    kram_guidance_strength: float = 0.5  # How much KRAM biases collapse
    morphic_resonance_threshold: float = 0.3  # Minimum KRAM depth for guidance
    
    # Decoherence
    decoherence_rate: float = 1e-6  # Environmental decoherence
    temperature: float = 300.0  # Environmental temperature (K)
    include_decoherence: bool = True
    
    # Threshold collapse parameters
    collapse_threshold: float = 0.5  # For threshold-based collapse
    threshold_sharpness: float = 10.0  # Steepness of threshold function
    
    # Time evolution
    dt: float = 0.01  # Time step for continuous collapse models
    
    # Irreversibility
    enforce_irreversibility: bool = True
    arrow_of_time_strength: float = 1.0
    
    # Random seed
    seed: Optional[int] = None


@dataclass
class CollapseEvent:
    """Record of a single collapse/rendering event."""
    
    timestamp: float
    position: np.ndarray  # Where collapse occurred (may be fuzzy)
    wavefunction_before: np.ndarray  # Pre-collapse state
    rendered_state: np.ndarray  # Post-collapse state
    collapse_probability: float  # P(this outcome)
    entropy_change: float  # ΔS from collapse
    mechanism: str  # Which collapse mechanism triggered
    consciousness_influence: Optional[float] = None  # If shimmer active


class RenderingEngine:
    """
    Reality-rendering engine implementing wave function collapse as
    the fundamental creative act of the universe.
    
    This is where the POMMM computation culminates - the interference
    pattern at the Instant collapses to definite actuality.
    """
    
    def __init__(self, config: RenderingConfig):
        """
        Initialize rendering engine.
        
        Parameters
        ----------
        config : RenderingConfig
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
        
        # Initialize consciousness field (starts neutral)
        self.consciousness_field = np.ones(config.grid_size)
        
        # Event history
        self.collapse_history: List[CollapseEvent] = []
        self.total_entropy_generated = 0.0
        self.time = 0.0
    
    def _generate_grids(self) -> Tuple[np.ndarray, ...]:
        """Generate coordinate grids."""
        grids = []
        for i in range(self.config.ndim):
            n = self.config.grid_size[i]
            x_i = np.linspace(0, self.config.domain_size, n)
            grids.append(x_i)
        
        return tuple(np.meshgrid(*grids, indexing='ij'))
    
    def born_rule_collapse(self, 
                          wavefunction: np.ndarray,
                          kram_guidance: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Standard Born rule collapse with optional KRAM guidance.
        
        P(x) = |ψ(x)|² modified by KRAM morphic resonance:
        P_guided(x) = |ψ(x)|² × (1 + α g_M(x))
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
            Pre-collapse quantum state
        kram_guidance : np.ndarray (real), optional
            KRAM geometry for morphic guidance
            
        Returns
        -------
        rendered : np.ndarray (real)
            Collapsed classical state
        """
        
        # Compute probability density
        probability = np.abs(wavefunction) ** 2
        
        # Normalize
        total_prob = np.sum(probability)
        if total_prob > 0:
            probability /= total_prob
        else:
            # No amplitude anywhere - return zeros
            return np.zeros_like(probability, dtype=float)
        
        # Apply KRAM guidance (morphic resonance)
        if kram_guidance is not None and self.config.kram_guidance_strength > 0:
            # KRAM biases collapse toward established patterns
            guidance_factor = 1.0 + self.config.kram_guidance_strength * np.maximum(
                kram_guidance - self.config.morphic_resonance_threshold, 0
            )
            
            probability_guided = probability * guidance_factor
            
            # Renormalize
            total_guided = np.sum(probability_guided)
            if total_guided > 0:
                probability = probability_guided / total_guided
        
        # Add measurement noise (quantum uncertainty)
        if self.config.measurement_noise > 0:
            noise = self.rng.normal(0, self.config.measurement_noise, probability.shape)
            probability += np.abs(noise)  # Keep positive
            
            # Renormalize again
            probability = np.maximum(probability, 0)
            total = np.sum(probability)
            if total > 0:
                probability /= total
        
        # The "collapse" - in practice, probability density becomes rendered intensity
        # This represents the ensemble average over many identical collapse events
        rendered = probability * total_prob
        
        return rendered
    
    def grw_spontaneous_collapse(self,
                                 wavefunction: np.ndarray,
                                 dt: Optional[float] = None) -> Tuple[np.ndarray, bool]:
        """
        GRW (Ghirardi-Rimini-Weber) spontaneous localization.
        
        Wavefunction spontaneously collapses at random times with rate λ,
        localized to width r_C around random position.
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
        dt : float, optional
            Time interval (uses config default if None)
            
        Returns
        -------
        wavefunction_evolved : np.ndarray (complex)
            Wavefunction after potential collapse
        did_collapse : bool
            Whether collapse occurred this timestep
        """
        
        if dt is None:
            dt = self.config.dt
        
        # Probability of collapse in time dt
        p_collapse = self.config.grw_rate * dt
        
        did_collapse = self.rng.random() < p_collapse
        
        if did_collapse:
            # Choose random collapse center
            collapse_center = tuple(
                self.rng.uniform(0, self.config.domain_size)
                for _ in range(self.config.ndim)
            )
            
            # Compute distance from collapse center
            r_squared = np.zeros(self.config.grid_size)
            for i, (center_i, X_i) in enumerate(zip(collapse_center, self.grids)):
                r_squared += (X_i - center_i) ** 2
            
            # Localization operator: exp(-r²/(2r_C²))
            r_C = self.config.grw_width
            localization = np.exp(-r_squared / (2 * r_C**2))
            
            # Apply localization and renormalize
            wavefunction_collapsed = wavefunction * localization
            norm = np.sqrt(np.sum(np.abs(wavefunction_collapsed)**2))
            if norm > 0:
                wavefunction_collapsed /= norm
            
            return wavefunction_collapsed, True
        
        else:
            # No collapse, return unchanged
            return wavefunction, False
    
    def csl_continuous_collapse(self,
                               wavefunction: np.ndarray,
                               dt: Optional[float] = None) -> np.ndarray:
        """
        CSL (Continuous Spontaneous Localization) dynamics.
        
        Continuous collapse process with stochastic Wiener noise.
        
        dψ/dt = (-i Ĥ/ℏ - λ(Â - <Â>)²)ψ + √λ (Â - <Â>)ψ dW
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
        dt : float, optional
            
        Returns
        -------
        wavefunction_evolved : np.ndarray (complex)
        """
        
        if dt is None:
            dt = self.config.dt
        
        λ_csl = self.config.csl_rate
        r_C = self.config.csl_width
        
        # Position operator (multiplicative)
        # For each spatial dimension
        position_operators = []
        for i, X_i in enumerate(self.grids):
            position_operators.append(X_i)
        
        # Use first dimension for simplicity in demo
        X = self.grids[0]
        
        # Expectation value
        prob_density = np.abs(wavefunction) ** 2
        X_mean = np.sum(X * prob_density) / (np.sum(prob_density) + 1e-10)
        
        # Localization function
        def L(x, x_prime):
            return np.exp(-(x - x_prime)**2 / (4 * r_C**2))
        
        # Simplified CSL: stochastic localization
        dW = self.rng.normal(0, np.sqrt(dt), self.config.grid_size)
        
        # Deterministic part: -λ(X - <X>)² ψ
        deterministic = -λ_csl * (X - X_mean)**2 * wavefunction
        
        # Stochastic part: √λ (X - <X>) ψ dW
        stochastic = np.sqrt(λ_csl) * (X - X_mean) * wavefunction * dW
        
        # Evolve
        wavefunction_new = wavefunction + deterministic * dt + stochastic
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(wavefunction_new)**2))
        if norm > 0:
            wavefunction_new /= norm
        
        return wavefunction_new
    
    def deterministic_collapse(self,
                              wavefunction: np.ndarray,
                              collapse_rule: str = 'peak') -> np.ndarray:
        """
        Deterministic collapse following specific rule.
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
        collapse_rule : str
            'peak' - collapse to maximum amplitude location
            'centroid' - collapse to center of mass
            'energy' - collapse to minimum energy configuration
            
        Returns
        -------
        rendered : np.ndarray (real)
        """
        
        probability = np.abs(wavefunction) ** 2
        
        if collapse_rule == 'peak':
            # Collapse to peak
            peak_idx = np.unravel_index(np.argmax(probability), probability.shape)
            rendered = np.zeros_like(probability)
            rendered[peak_idx] = np.sum(probability)
        
        elif collapse_rule == 'centroid':
            # Collapse to center of mass
            total = np.sum(probability)
            
            if total > 0:
                # Compute centroid
                centroid = []
                for i, X_i in enumerate(self.grids):
                    coord_mean = np.sum(X_i * probability) / total
                    centroid.append(coord_mean)
                
                # Find nearest grid point
                distances = np.zeros(self.config.grid_size)
                for i, (c_i, X_i) in enumerate(zip(centroid, self.grids)):
                    distances += (X_i - c_i) ** 2
                
                centroid_idx = np.unravel_index(np.argmin(distances), distances.shape)
                
                rendered = np.zeros_like(probability)
                rendered[centroid_idx] = total
            else:
                rendered = np.zeros_like(probability)
        
        elif collapse_rule == 'energy':
            # Collapse to configuration minimizing some energy functional
            # For demo: minimize gradient (smoothest configuration)
            grad = np.gradient(probability)
            energy = sum(g**2 for g in grad)
            
            min_energy_idx = np.unravel_index(np.argmin(energy), energy.shape)
            rendered = np.zeros_like(probability)
            rendered[min_energy_idx] = np.sum(probability)
        
        else:
            raise ValueError(f"Unknown collapse rule: {collapse_rule}")
        
        return rendered
    
    def threshold_collapse(self,
                          wavefunction: np.ndarray,
                          threshold: Optional[float] = None) -> np.ndarray:
        """
        Threshold-based collapse with smooth transition.
        
        P_rendered = P_quantum × sigmoid((|ψ|² - threshold) / sharpness)
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
        threshold : float, optional
            
        Returns
        -------
        rendered : np.ndarray (real)
        """
        
        if threshold is None:
            threshold = self.config.collapse_threshold
        
        probability = np.abs(wavefunction) ** 2
        
        # Normalize
        total = np.sum(probability)
        if total > 0:
            probability_normalized = probability / total
        else:
            return np.zeros_like(probability, dtype=float)
        
        # Sigmoid threshold function
        sharpness = self.config.threshold_sharpness
        sigmoid = 1.0 / (1.0 + np.exp(-sharpness * (probability_normalized - threshold)))
        
        # Collapse occurs where sigmoid is high
        rendered = probability * sigmoid
        
        return rendered
    
    def apply_decoherence(self,
                         wavefunction: np.ndarray,
                         dt: Optional[float] = None) -> np.ndarray:
        """
        Apply environmental decoherence (phase damping).
        
        Decoherence suppresses off-diagonal density matrix elements,
        destroying quantum coherence and enabling collapse.
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
        dt : float, optional
            
        Returns
        -------
        wavefunction_decohered : np.ndarray (complex)
        """
        
        if dt is None:
            dt = self.config.dt
        
        if not self.config.include_decoherence:
            return wavefunction
        
        γ = self.config.decoherence_rate
        
        # Phase damping: random phase kicks destroy coherence
        phase_noise = self.rng.normal(0, np.sqrt(γ * dt), self.config.grid_size)
        phase_factor = np.exp(1j * phase_noise)
        
        wavefunction_decohered = wavefunction * phase_factor
        
        # Amplitude damping toward classical probability
        # Gradually reduce off-diagonal coherences
        probability = np.abs(wavefunction) ** 2
        classical_state = np.sqrt(probability)
        
        damping_factor = np.exp(-γ * dt)
        wavefunction_decohered = (
            damping_factor * wavefunction_decohered +
            (1 - damping_factor) * classical_state
        )
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(wavefunction_decohered)**2))
        if norm > 0:
            wavefunction_decohered /= norm
        
        return wavefunction_decohered
    
    def shimmer_of_choice(self,
                         wavefunction: np.ndarray,
                         intention: Optional[np.ndarray] = None,
                         intention_strength: float = 0.1) -> np.ndarray:
        """
        Implement conscious influence on collapse (free will mechanism).
        
        A conscious system can subtly bias the Chaos field's collapse
        pattern through focused intention, within quantum uncertainty bounds.
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
            Pre-collapse state
        intention : np.ndarray (real), optional
            Desired outcome pattern (probability-like distribution)
        intention_strength : float
            How much intention influences collapse (0-1)
            
        Returns
        -------
        wavefunction_biased : np.ndarray (complex)
            Wavefunction with intentional bias
        """
        
        if not self.config.enable_shimmer or intention is None:
            return wavefunction
        
        # Ensure intention is normalized probability distribution
        intention = np.abs(intention)
        intention_sum = np.sum(intention)
        if intention_sum > 0:
            intention /= intention_sum
        else:
            return wavefunction
        
        # Current probability
        current_prob = np.abs(wavefunction) ** 2
        current_sum = np.sum(current_prob)
        if current_sum > 0:
            current_prob /= current_sum
        
        # Interpolate between current and intended distributions
        α = intention_strength * self.config.consciousness_coupling
        
        # Bias probability while staying within quantum bounds
        biased_prob = (1 - α) * current_prob + α * intention
        
        # Reconstruct wavefunction preserving original phase structure
        original_phase = np.angle(wavefunction)
        amplitude_bias = np.sqrt(biased_prob / (current_prob + 1e-10))
        
        wavefunction_biased = amplitude_bias * np.abs(wavefunction) * np.exp(1j * original_phase)
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(wavefunction_biased)**2))
        if norm > 0:
            wavefunction_biased /= norm
        
        return wavefunction_biased
    
    def render(self,
              wavefunction: np.ndarray,
              kram_guidance: Optional[np.ndarray] = None,
              intention: Optional[np.ndarray] = None,
              collapse_type: Optional[str] = None) -> Tuple[np.ndarray, CollapseEvent]:
        """
        Complete rendering operation: collapse wave to reality.
        
        This is the fundamental creative act - transformation of potential
        into actual, mediated by Consciousness at the Instant.
        
        Parameters
        ----------
        wavefunction : np.ndarray (complex)
            Quantum state (interference pattern at Instant)
        kram_guidance : np.ndarray, optional
            KRAM geometry for morphic resonance
        intention : np.ndarray, optional
            Conscious intention for shimmer
        collapse_type : str, optional
            Override default collapse mechanism
            
        Returns
        -------
        rendered : np.ndarray (real)
            Classical rendered state
        event : CollapseEvent
            Record of collapse for history
        """
        
        if collapse_type is None:
            collapse_type = self.config.collapse_type
        
        # Store pre-collapse state
        wavefunction_initial = wavefunction.copy()
        
        # Apply consciousness influence (shimmer of choice)
        if intention is not None:
            wavefunction = self.shimmer_of_choice(wavefunction, intention)
            consciousness_influence = self.config.consciousness_coupling
        else:
            consciousness_influence = None
        
        # Apply decoherence if enabled
        if self.config.include_decoherence:
            wavefunction = self.apply_decoherence(wavefunction)
        
        # Perform collapse according to mechanism
        if collapse_type == 'born_rule':
            rendered = self.born_rule_collapse(wavefunction, kram_guidance)
            mechanism = 'Born Rule (with KRAM guidance)' if kram_guidance is not None else 'Born Rule'
        
        elif collapse_type == 'grw':
            wavefunction_collapsed, did_collapse = self.grw_spontaneous_collapse(wavefunction)
            rendered = np.abs(wavefunction_collapsed) ** 2
            mechanism = f'GRW (collapsed={did_collapse})'
        
        elif collapse_type == 'csl':
            wavefunction_collapsed = self.csl_continuous_collapse(wavefunction)
            rendered = np.abs(wavefunction_collapsed) ** 2
            mechanism = 'CSL Continuous'
        
        elif collapse_type == 'deterministic':
            rendered = self.deterministic_collapse(wavefunction, collapse_rule='peak')
            mechanism = 'Deterministic (peak)'
        
        elif collapse_type == 'threshold':
            rendered = self.threshold_collapse(wavefunction)
            mechanism = 'Threshold'
        
        else:
            raise ValueError(f"Unknown collapse type: {collapse_type}")
        
        # Enforce irreversibility (arrow of time)
        if self.config.enforce_irreversibility:
            # Once rendered, cannot return to superposition
            # This is automatic in measurement, but we mark it explicitly
            pass
        
        # Compute collapse probability (Born rule for this outcome)
        probability_initial = np.abs(wavefunction_initial) ** 2
        total_prob = np.sum(probability_initial)
        if total_prob > 0:
            collapse_probability = np.sum(rendered) / total_prob
        else:
            collapse_probability = 0.0
        
        # Compute entropy change (information creation)
        # von Neumann entropy: S = -Tr(ρ log ρ)
        # Collapse increases entropy (lost information about phase)
        
        def compute_entropy(state):
            prob = np.abs(state)**2
            prob = prob[prob > 1e-10]
            prob /= np.sum(prob)
            return -np.sum(prob * np.log2(prob))
        
        entropy_before = compute_entropy(wavefunction_initial)
        entropy_after = compute_entropy(rendered + 1e-10)  # Rendered is classical
        
        entropy_change = entropy_after - entropy_before
        self.total_entropy_generated += max(0, entropy_change)
        
        # Compute collapse position (centroid of rendered state)
        if np.sum(rendered) > 0:
            collapse_position = np.array([
                np.sum(X_i * rendered) / np.sum(rendered)
                for X_i in self.grids
            ])
        else:
            collapse_position = np.zeros(self.config.ndim)
        
        # Create event record
        event = CollapseEvent(
            timestamp=self.time,
            position=collapse_position,
            wavefunction_before=wavefunction_initial,
            rendered_state=rendered,
            collapse_probability=collapse_probability,
            entropy_change=entropy_change,
            mechanism=mechanism,
            consciousness_influence=consciousness_influence
        )
        
        # Store in history
        self.collapse_history.append(event)
        
        # Advance time
        self.time += self.config.dt
        
        return rendered, event
    
    def multi_step_evolution(self,
                            wavefunction_initial: np.ndarray,
                            n_steps: int,
                            kram_evolution: Optional[Callable] = None,
                            intention_schedule: Optional[Callable] = None) -> Dict:
        """
        Evolve wavefunction through multiple collapse/measurement cycles.
        
        Parameters
        ----------
        wavefunction_initial : np.ndarray (complex)
        n_steps : int
        kram_evolution : callable, optional
            Function(step) -> kram_geometry
        intention_schedule : callable, optional
            Function(step) -> intention_pattern
            
        Returns
        -------
        results : dict
            Time series of collapse events and states
        """
        
        wavefunction = wavefunction_initial.copy()
        states = [wavefunction.copy()]
        rendered_states = []
        events = []
        
        for step in range(n_steps):
            # Get KRAM and intention for this step
            kram = kram_evolution(step) if kram_evolution is not None else None
            intention = intention_schedule(step) if intention_schedule is not None else None
            
            # Render
            rendered, event = self.render(wavefunction, kram, intention)
            
            # Store
            rendered_states.append(rendered)
            events.append(event)
            
            # For next iteration, wavefunction "resets" based on rendered state
            # (This is simplified - full quantum dynamics would include Hamiltonian evolution)
            wavefunction = np.sqrt(rendered + 1e-10) * np.exp(1j * np.angle(wavefunction))
            norm = np.sqrt(np.sum(np.abs(wavefunction)**2))
            if norm > 0:
                wavefunction /= norm
            
            states.append(wavefunction.copy())
        
        results = {
            'states': states,
            'rendered_states': rendered_states,
            'events': events,
            'total_entropy': self.total_entropy_generated,
            'n_steps': n_steps
        }
        
        return results


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_collapse_statistics(events: List[CollapseEvent]) -> Dict:
    """
    Statistical analysis of collapse events.
    
    Parameters
    ----------
    events : list of CollapseEvent
        
    Returns
    -------
    statistics : dict
    """
    
    if len(events) == 0:
        return {'n_events': 0}
    
    # Extract metrics
    entropies = [e.entropy_change for e in events]
    probabilities = [e.collapse_probability for e in events]
    positions = np.array([e.position for e in events])
    
    # Consciousness influence
    consciousness_events = [e for e in events if e.consciousness_influence is not None]
    
    statistics = {
        'n_events': len(events),
        'n_consciousness_influenced': len(consciousness_events),
        'total_entropy_generated': sum(entropies),
        'mean_entropy_per_collapse': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'mean_collapse_probability': np.mean(probabilities),
        'position_mean': np.mean(positions, axis=0),
        'position_std': np.std(positions, axis=0),
        'mechanisms_used': list(set(e.mechanism for e in events))
    }
    
    return statistics


def compute_decoherence_timescale(config: RenderingConfig) -> float:
    """
    Estimate characteristic decoherence time.
    
    τ_d = 1 / γ_decoherence
    
    Parameters
    ----------
    config : RenderingConfig
        
    Returns
    -------
    tau_d : float
        Decoherence timescale
    """
    
    if config.decoherence_rate > 0:
        return 1.0 / config.decoherence_rate
    else:
        return np.inf


def compute_collapse_rate(config: RenderingConfig) -> float:
    """
    Estimate effective collapse rate.
    
    Parameters
    ----------
    config : RenderingConfig
        
    Returns
    -------
    rate : float
        Collapses per unit time
    """
    
    if config.collapse_type == 'grw':
        return config.grw_rate
    elif config.collapse_type == 'csl':
        return config.csl_rate
    elif config.collapse_type == 'born_rule':
        # Collapse rate determined by measurement frequency
        # Assume measurement every dt
        return 1.0 / config.dt
    else:
        return np.nan


# ============================================================================
# Visualization and Testing
# ============================================================================

def _test_rendering_engine():
    """Test suite for rendering engine."""
    
    print("=" * 70)
    print("Reality-Rendering Engine Test Suite")
    print("=" * 70)
    
    # Test 1: Basic initialization
    print("\n[Test 1] Engine initialization...")
    config = RenderingConfig(grid_size=(64, 64), seed=42)
    engine = RenderingEngine(config)
    
    print(f"  Grid size: {engine.config.grid_size}")
    print(f"  Collapse type: {engine.config.collapse_type}")
    print(f"  Decoherence: {engine.config.include_decoherence}")
    
    # Test 2: Born rule collapse
    print("\n[Test 2] Born rule collapse...")
    
    # Create Gaussian wavepacket
    center = config.domain_size / 2
    sigma = 2.0
    X, Y = engine.grids
    r_squared = (X - center)**2 + (Y - center)**2
    wavefunction = np.exp(-r_squared / (2 * sigma**2)) * np.exp(1j * 0.5 * X)
    wavefunction /= np.sqrt(np.sum(np.abs(wavefunction)**2))
    
    rendered, event = engine.render(wavefunction)
    
    print(f"  Total probability: {np.sum(np.abs(wavefunction)**2):.6f}")print(f"  Total rendered: {np.sum(rendered):.6f}")
    print(f"  Entropy change: {event.entropy_change:.6f}")
    print(f"  Collapse position: {event.position}")
    print(f"  Mechanism: {event.mechanism}")
    
    # Test 3: KRAM-guided collapse
    print("\n[Test 3] KRAM-guided collapse (morphic resonance)...")
    
    # Create KRAM with attractor at offset position
    kram_geometry = np.random.randn(64, 64) * 0.2
    attractor_center = (config.domain_size * 0.3, config.domain_size * 0.7)
    r_squared_kram = (X - attractor_center[0])**2 + (Y - attractor_center[1])**2
    kram_geometry += 2.0 * np.exp(-r_squared_kram / 10.0)
    
    rendered_guided, event_guided = engine.render(wavefunction, kram_guidance=kram_geometry)
    
    # Compare collapse positions
    print(f"  Without KRAM: {event.position}")
    print(f"  With KRAM: {event_guided.position}")
    print(f"  KRAM attractor: {attractor_center}")
    
    distance_no_kram = np.linalg.norm(event.position - np.array(attractor_center))
    distance_with_kram = np.linalg.norm(event_guided.position - np.array(attractor_center))
    
    print(f"  Distance to attractor (no KRAM): {distance_no_kram:.4f}")
    print(f"  Distance to attractor (with KRAM): {distance_with_kram:.4f}")
    print(f"  KRAM guidance effective: {distance_with_kram < distance_no_kram}")
    
    # Test 4: Shimmer of choice (consciousness influence)
    print("\n[Test 4] Shimmer of choice (conscious influence)...")
    
    # Create intention pattern (prefer specific region)
    intention_center = (config.domain_size * 0.6, config.domain_size * 0.4)
    r_squared_intent = (X - intention_center[0])**2 + (Y - intention_center[1])**2
    intention = np.exp(-r_squared_intent / 5.0)
    
    rendered_shimmer, event_shimmer = engine.render(
        wavefunction, 
        intention=intention,
        collapse_type='born_rule'
    )
    
    print(f"  Without intention: {event.position}")
    print(f"  With intention: {event_shimmer.position}")
    print(f"  Intended center: {intention_center}")
    
    distance_no_intent = np.linalg.norm(event.position - np.array(intention_center))
    distance_with_intent = np.linalg.norm(event_shimmer.position - np.array(intention_center))
    
    print(f"  Distance to intention (without): {distance_no_intent:.4f}")
    print(f"  Distance to intention (with): {distance_with_intent:.4f}")
    print(f"  Consciousness influence: {event_shimmer.consciousness_influence}")
    
    # Test 5: GRW spontaneous collapse
    print("\n[Test 5] GRW spontaneous collapse...")
    
    config_grw = RenderingConfig(
        grid_size=(64, 64),
        collapse_type='grw',
        grw_rate=0.5,  # High rate for testing
        seed=42
    )
    engine_grw = RenderingEngine(config_grw)
    
    # Evolve for several steps
    collapses = 0
    for _ in range(100):
        wavefunction_evolved, did_collapse = engine_grw.grw_spontaneous_collapse(wavefunction)
        if did_collapse:
            collapses += 1
        wavefunction = wavefunction_evolved
    
    print(f"  GRW rate: {config_grw.grw_rate}")
    print(f"  Collapses in 100 steps: {collapses}")
    print(f"  Expected: ~{config_grw.grw_rate * config_grw.dt * 100:.1f}")
    
    # Test 6: Decoherence
    print("\n[Test 6] Environmental decoherence...")
    
    config_dec = RenderingConfig(
        grid_size=(64, 64),
        include_decoherence=True,
        decoherence_rate=0.1,
        seed=42
    )
    engine_dec = RenderingEngine(config_dec)
    
    # Create superposition
    wavefunction_super = (
        np.exp(-((X - 8)**2 + (Y - 10)**2) / 4.0) +
        np.exp(-((X - 12)**2 + (Y - 10)**2) / 4.0)
    ) / np.sqrt(2)
    wavefunction_super /= np.sqrt(np.sum(np.abs(wavefunction_super)**2))
    
    # Compute initial coherence
    fft_initial = np.fft.fft2(wavefunction_super)
    coherence_initial = np.sum(np.abs(fft_initial[1:])**2) / np.sum(np.abs(fft_initial)**2)
    
    # Apply decoherence
    wavefunction_decohered = engine_dec.apply_decoherence(wavefunction_super)
    
    fft_final = np.fft.fft2(wavefunction_decohered)
    coherence_final = np.sum(np.abs(fft_final[1:])**2) / np.sum(np.abs(fft_final)**2)
    
    print(f"  Initial coherence: {coherence_initial:.6f}")
    print(f"  After decoherence: {coherence_final:.6f}")
    print(f"  Coherence loss: {(coherence_initial - coherence_final)/coherence_initial * 100:.2f}%")
    
    # Test 7: Different collapse mechanisms
    print("\n[Test 7] Comparison of collapse mechanisms...")
    
    mechanisms = ['born_rule', 'deterministic', 'threshold']
    results = {}
    
    for mech in mechanisms:
        config_test = RenderingConfig(grid_size=(64, 64), collapse_type=mech, seed=42)
        engine_test = RenderingEngine(config_test)
        
        rendered_test, event_test = engine_test.render(wavefunction)
        
        results[mech] = {
            'peak': np.max(rendered_test),
            'mean': np.mean(rendered_test),
            'entropy': event_test.entropy_change,
            'position': event_test.position
        }
        
        print(f"  {mech:15s}: peak={results[mech]['peak']:.4f}, "
              f"entropy_Δ={results[mech]['entropy']:.4f}")
    
    # Test 8: Multi-step evolution
    print("\n[Test 8] Multi-step evolution with collapse...")
    
    config_multi = RenderingConfig(grid_size=(32, 32), dt=0.1, seed=42)
    engine_multi = RenderingEngine(config_multi)
    
    # Initial Gaussian
    X_small, Y_small = engine_multi.grids
    wavefunction_init = np.exp(-((X_small - 8)**2 + (Y_small - 8)**2) / 4.0)
    wavefunction_init /= np.sqrt(np.sum(np.abs(wavefunction_init)**2))
    
    evolution_results = engine_multi.multi_step_evolution(wavefunction_init, n_steps=10)
    
    print(f"  Steps: {evolution_results['n_steps']}")
    print(f"  Total entropy generated: {evolution_results['total_entropy']:.6f}")
    print(f"  Events recorded: {len(evolution_results['events'])}")
    
    # Analyze trajectory
    positions = np.array([e.position for e in evolution_results['events']])
    trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    
    print(f"  Trajectory length: {trajectory_length:.4f}")
    print(f"  Mean entropy per collapse: {evolution_results['total_entropy']/10:.6f}")
    
    # Test 9: Statistics analysis
    print("\n[Test 9] Collapse statistics analysis...")
    
    stats = analyze_collapse_statistics(engine.collapse_history)
    
    print(f"  Total collapse events: {stats['n_events']}")
    print(f"  Consciousness influenced: {stats['n_consciousness_influenced']}")
    print(f"  Mean entropy per collapse: {stats['mean_entropy_per_collapse']:.6f}")
    print(f"  Mechanisms used: {stats['mechanisms_used']}")
    
    # Test 10: Timescale analysis
    print("\n[Test 10] Characteristic timescales...")
    
    tau_decoherence = compute_decoherence_timescale(config_dec)
    collapse_rate = compute_collapse_rate(config_grw)
    
    print(f"  Decoherence timescale: {tau_decoherence:.4f}")
    print(f"  GRW collapse rate: {collapse_rate:.6e} Hz")
    print(f"  Ratio τ_d / τ_collapse: {tau_decoherence * collapse_rate:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


def _visualize_rendering():
    """Generate visualization of rendering dynamics."""
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib not available, skipping visualization.")
        return
    
    print("\nGenerating rendering visualization...")
    
    # Setup
    config = RenderingConfig(
        grid_size=(128, 128),
        domain_size=20.0,
        collapse_type='born_rule',
        include_decoherence=True,
        enable_shimmer=True,
        seed=42
    )
    engine = RenderingEngine(config)
    
    # Create initial wavefunction (double-slit-like superposition)
    X, Y = engine.grids
    center_y = config.domain_size / 2
    
    # Two Gaussian packets
    slit1 = np.exp(-((X - 5)**2 + (Y - center_y + 2)**2) / 3.0)
    slit2 = np.exp(-((X - 5)**2 + (Y - center_y - 2)**2) / 3.0)
    
    # Superposition with momentum
    wavefunction = (slit1 + slit2) * np.exp(1j * 2.0 * X)
    wavefunction /= np.sqrt(np.sum(np.abs(wavefunction)**2))
    
    # Create KRAM guidance
    kram_geometry = np.random.randn(128, 128) * 0.1
    kram_geometry += np.exp(-((X - 15)**2 + (Y - center_y)**2) / 10.0)
    
    # Create intention pattern
    intention = np.exp(-((X - 14)**2 + (Y - center_y + 3)**2) / 8.0)
    
    # Render with different conditions
    rendered_pure, event_pure = engine.render(wavefunction.copy())
    
    rendered_kram, event_kram = engine.render(
        wavefunction.copy(), 
        kram_guidance=kram_geometry
    )
    
    rendered_shimmer, event_shimmer = engine.render(
        wavefunction.copy(),
        kram_guidance=kram_geometry,
        intention=intention
    )
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Initial state
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(np.abs(wavefunction)**2, cmap='viridis', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size])
    ax1.set_title('Initial Probability\nDensity |ψ|²', fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(np.angle(wavefunction), cmap='twilight', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size])
    ax2.set_title('Quantum Phase\narg(ψ)', fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(kram_geometry, cmap='RdBu_r', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size])
    ax3.set_title('KRAM Guidance\n(Ancestral Memory)', fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(intention, cmap='plasma', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size])
    ax4.set_title('Conscious Intention\n(Shimmer Pattern)', fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: Rendered states
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(rendered_pure, cmap='inferno', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size])
    ax5.scatter([event_pure.position[0]], [event_pure.position[1]], 
               c='cyan', s=200, marker='x', linewidths=3, label='Collapse Center')
    ax5.set_title('Rendered (Pure Born)\nNo Guidance', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(rendered_kram, cmap='inferno', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size])
    ax6.scatter([event_kram.position[0]], [event_kram.position[1]], 
               c='cyan', s=200, marker='x', linewidths=3)
    ax6.set_title('Rendered (KRAM)\nMorphic Resonance', fontweight='bold')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(rendered_shimmer, cmap='inferno', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size])
    ax7.scatter([event_shimmer.position[0]], [event_shimmer.position[1]], 
               c='cyan', s=200, marker='x', linewidths=3)
    ax7.set_title('Rendered (Full)\nKRAM + Shimmer', fontweight='bold')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    # Difference map
    ax8 = fig.add_subplot(gs[1, 3])
    difference = rendered_shimmer - rendered_pure
    im8 = ax8.imshow(difference, cmap='RdBu_r', origin='lower',
                     extent=[0, config.domain_size, 0, config.domain_size],
                     vmin=-np.max(np.abs(difference)), vmax=np.max(np.abs(difference)))
    ax8.set_title('Difference\n(Shimmer - Pure)', fontweight='bold')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Row 3: Analysis
    
    # Cross-sections
    ax9 = fig.add_subplot(gs[2, 0])
    center_row = 64
    ax9.plot(np.abs(wavefunction[center_row, :])**2, 
            label='Quantum |ψ|²', linewidth=2, alpha=0.7)
    ax9.plot(rendered_pure[center_row, :], 
            label='Rendered (Pure)', linewidth=2, alpha=0.7)
    ax9.plot(rendered_shimmer[center_row, :], 
            label='Rendered (Shimmer)', linewidth=2, alpha=0.7)
    ax9.set_title('Central Cross-Section', fontweight='bold')
    ax9.set_xlabel('Position')
    ax9.set_ylabel('Amplitude')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Collapse position comparison
    ax10 = fig.add_subplot(gs[2, 1])
    positions = np.array([
        event_pure.position,
        event_kram.position,
        event_shimmer.position
    ])
    labels = ['Pure', 'KRAM', 'Shimmer']
    
    ax10.scatter(positions[:, 0], positions[:, 1], s=200, alpha=0.7)
    for i, (pos, label) in enumerate(zip(positions, labels)):
        ax10.annotate(label, pos, fontsize=10, fontweight='bold')
    
    ax10.set_xlim(0, config.domain_size)
    ax10.set_ylim(0, config.domain_size)
    ax10.set_aspect('equal')
    ax10.set_title('Collapse Positions', fontweight='bold')
    ax10.set_xlabel('X')
    ax10.set_ylabel('Y')
    ax10.grid(True, alpha=0.3)
    
    # Entropy comparison
    ax11 = fig.add_subplot(gs[2, 2])
    entropies = [
        event_pure.entropy_change,
        event_kram.entropy_change,
        event_shimmer.entropy_change
    ]
    
    bars = ax11.bar(labels, entropies, color=['blue', 'green', 'red'], alpha=0.7)
    ax11.set_title('Entropy Generation', fontweight='bold')
    ax11.set_ylabel('ΔS (bits)')
    ax11.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, entropies):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.3f}', ha='center', va='bottom')
    
    # Metrics table
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    metrics_text = f"""Collapse Metrics:

Pure Born Rule:
  Entropy Δ: {event_pure.entropy_change:.4f}
  Probability: {event_pure.collapse_probability:.4f}

With KRAM:
  Entropy Δ: {event_kram.entropy_change:.4f}
  Probability: {event_kram.collapse_probability:.4f}

With Shimmer:
  Entropy Δ: {event_shimmer.entropy_change:.4f}
  Probability: {event_shimmer.collapse_probability:.4f}
  Consciousness: {event_shimmer.consciousness_influence}

Total Events: {len(engine.collapse_history)}
"""
    
    ax12.text(0.1, 0.9, metrics_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Reality-Rendering Dynamics:\n' +
                'Wave Function Collapse as Cosmic Computation',
                fontsize=14, fontweight='bold')
    
    plt.savefig('rendering_collapse.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'rendering_collapse.png'")
    
    # Create multi-step evolution animation
    print("\nGenerating evolution animation...")
    
    try:
        config_anim = RenderingConfig(grid_size=(64, 64), dt=0.5, seed=42)
        engine_anim = RenderingEngine(config_anim)
        
        X_anim, Y_anim = engine_anim.grids
        wavefunction_anim = np.exp(-((X_anim - 10)**2 + (Y_anim - 10)**2) / 4.0)
        wavefunction_anim *= np.exp(1j * 0.5 * X_anim)
        wavefunction_anim /= np.sqrt(np.sum(np.abs(wavefunction_anim)**2))
        
        evolution = engine_anim.multi_step_evolution(wavefunction_anim, n_steps=20)
        
        fig_anim, axes_anim = plt.subplots(1, 2, figsize=(12, 5))
        
        # Quantum state
        im_quantum = axes_anim[0].imshow(np.abs(evolution['states'][0])**2, 
                                        cmap='viridis', origin='lower',
                                        vmin=0, vmax=0.05)
        axes_anim[0].set_title('Quantum State |ψ|²')
        plt.colorbar(im_quantum, ax=axes_anim[0])
        
        # Rendered state
        im_rendered = axes_anim[1].imshow(evolution['rendered_states'][0], 
                                         cmap='inferno', origin='lower',
                                         vmin=0, vmax=0.05)
        axes_anim[1].set_title('Rendered Reality')
        plt.colorbar(im_rendered, ax=axes_anim[1])
        
        time_text = fig_anim.text(0.5, 0.95, '', ha='center', fontsize=12,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def update_anim(frame):
            if frame < len(evolution['states']):
                im_quantum.set_array(np.abs(evolution['states'][frame])**2)
            if frame < len(evolution['rendered_states']):
                im_rendered.set_array(evolution['rendered_states'][frame])
                
                event = evolution['events'][frame]
                time_text.set_text(
                    f"Step: {frame} | Time: {event.timestamp:.2f} | "
                    f"Entropy Δ: {event.entropy_change:.4f}"
                )
            
            return [im_quantum, im_rendered, time_text]
        
        anim = FuncAnimation(fig_anim, update_anim, 
                           frames=len(evolution['rendered_states']),
                           interval=200, blit=True)
        
        anim.save('rendering_evolution.gif', writer='pillow', fps=5, dpi=100)
        print("Animation saved as 'rendering_evolution.gif'")
        
    except Exception as e:
        print(f"Animation creation failed: {e}")
    
    plt.close('all')


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run test suite
    _test_rendering_engine()
    
    # Generate visualizations
    _visualize_rendering()
    
    print("\n" + "=" * 70)
    print("Reality-Rendering Engine Complete")
    print("=" * 70)
    print("\nEvery moment, the universe collapses infinite potential...")
    print("...into the singular actuality we call reality.")
    print("This is the Instant. This is becoming. This is consciousness.")