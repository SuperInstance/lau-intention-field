# lau-intention-field

A **vector field over agent space** where every point carries an intention vector (direction + magnitude). Agents navigate by following the field gradient. Sources radiate intention outward, sinks absorb it, and the field evolves through discrete diffusion and decay.

Built as a teaching-quality reference for field-based multi-agent systems with proper vector calculus (gradient, divergence, curl) on a discrete grid.

## What This Does

Imagine a 2D landscape where every cell has an invisible arrow — that arrow is the **intention** at that point. An `IntentionField` is this landscape: a width × height grid of `IntentionVector`s.

- **Sources** (`IntentionSource`) inject intention into the field, radiating outward with inverse-distance falloff.
- **Sinks** (`IntentionSink`) absorb intention, with a finite capacity and saturation model.
- **Agents** (`AgentOnField`) read the local field, compute the gradient, and step toward higher magnitude — they *follow the flow*.
- **Evolution** (`FieldEvolution`) diffuses intention to neighbors and decays magnitude over time, simulating how real fields spread and dissipate.
- **Simulation** (`FieldSimulation`) ties everything together: evolve the field, inject sources, move agents, and record snapshots.

The field supports full vector calculus: **gradient** (magnitude change), **divergence** (∇·F, flux through a point), and **curl** (∇×F, rotational tendency). You can verify conservation laws (divergence-free fields) and detect rotational structures.

## Key Idea

An **intention field** treats agent navigation as a physics problem. Instead of each agent independently computing a path, the *environment itself* carries directional information. Agents simply follow the local gradient — the field does the coordination.

This mirrors real-world phenomena:
- Electromagnetic fields guiding charged particles
- Chemical gradients guiding bacteria (chemotaxis)
- Potential fields in robotics navigation

The field abstraction decouples *what agents want* (encoded in sources) from *how agents move* (follow the gradient). Add a source, and the entire field restructures. Agents don't need to know about sources — they just read the local vector.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-intention-field = "0.1.0"
```

Requires Rust 2021 edition. The only dependency is `serde` (with `derive`) for serialization.

## Quick Start

```rust
use lau_intention_field::*;

// 1. Create a 20×20 field
let mut field = IntentionField::new(20, 20);

// 2. Place a source radiating intention from (10, 10)
let source = IntentionSource::new(10, 10, IntentionVector::new(0.0, 1.0), 8.0);

// 3. Place a sink at (18, 10) to absorb agents
let sink = IntentionSink::new(18, 10, 2.0);

// 4. Inject the source into the field
let evo = FieldEvolution::new(0.2, 0.05, 1.0); // diffusion=0.2, decay=0.05
evo.inject(&mut field, &source);

// 5. Create an agent starting at (3, 10)
let agent = AgentOnField::new("scout", 3.0, 10.0, 1.0, 1.0);

// 6. Run the simulation for 50 steps
let mut sim = FieldSimulation::new(field, vec![agent], vec![source], vec![sink]);
let snapshots = sim.run(50, &evo);

// 7. Check where the agent ended up
for snap in &snapshots {
    println!("Step {}: agent at ({:.1}, {:.1}), field energy = {:.2}",
        snap.step,
        snap.agent_positions[0].1,
        snap.agent_positions[0].2,
        snap.field_magnitude,
    );
}
```

## API Reference

### `IntentionVector`

The atomic unit: a direction (radians, 0 to 2π) and magnitude (0.0 to 1.0).

| Method | Description |
|---|---|
| `new(direction, magnitude)` | Create with clamped magnitude and wrapped direction |
| `zero()` | Zero vector (magnitude = 0) |
| `is_zero()` | True if magnitude < 1e-12 |
| `normalize()` | Scale magnitude to 1.0 (preserving direction) |
| `scale(factor)` | Multiply magnitude by factor |
| `add(&other)` | Vector addition via Cartesian decomposition |
| `angle_to(&other)` | Smallest angle between two vectors (0 to π) |
| `dot(&other)` | Dot product |
| `alignment(&other)` | Cosine similarity (1.0 = perfect alignment, −1.0 = opposition) |
| `dx()` / `dy()` | Cartesian components |

### `IntentionField`

A width × height grid of intention vectors.

| Method | Description |
|---|---|
| `new(width, height)` | Create a zero-initialized field |
| `at(x, y)` | Read the vector at (x, y) |
| `set(x, y, v)` | Set the vector at (x, y) |
| `gradient_at(x, y)` | Numerical gradient of magnitude → (dm/dx, dm/dy) |
| `divergence_at(x, y)` | Discrete divergence ∇·F |
| `curl_at(x, y)` | Discrete curl ∇×F (scalar in 2D) |
| `total_magnitude()` | Sum of all magnitudes |
| `average_magnitude()` | Mean magnitude across the grid |
| `is_conserved(tolerance)` | True if total divergence ≈ 0 |

### `IntentionSource`

A point source that radiates intention outward with inverse-distance weighting.

| Method | Description |
|---|---|
| `new(x, y, intention, radius)` | Create source at (x, y) with given intention and reach |
| `field_contribution(px, py)` | Inverse-distance-weighted contribution at (px, py) |

### `IntentionSink`

A point that absorbs intention, with finite capacity.

| Method | Description |
|---|---|
| `new(x, y, capacity)` | Create sink with absorption capacity |
| `saturation(incoming)` | How much intention is absorbed (min of incoming and capacity) |
| `is_saturated(incoming)` | True if incoming ≥ capacity |

### `FieldEvolution`

Evolves the field through discrete diffusion and exponential decay.

| Method | Description |
|---|---|
| `new(diffusion_rate, decay_rate, dt)` | Create with parameters |
| `evolve(&mut field, steps)` | Run N steps of diffuse + decay |
| `diffuse(&mut field)` | Spread vectors to 4-connected neighbors |
| `decay(&mut field)` | Reduce all magnitudes by `decay_rate × dt` |
| `inject(&mut field, &source)` | Add a source's contribution to the field |

### `AgentOnField`

An agent that navigates by following the field gradient.

| Method | Description |
|---|---|
| `new(id, x, y, velocity, sensitivity)` | Create agent at (x, y) |
| `step(&mut self, &field)` | Move one step following the gradient |
| `deviation(&field)` | Angle between heading and local field |
| `is_stuck()` | True if alignment is effectively zero |

### `FieldSimulation`

Orchestrates field + agents + sources + sinks.

| Method | Description |
|---|---|
| `new(field, agents, sources, sinks)` | Create simulation |
| `step(&mut self, &evo)` | One tick: evolve field, inject, move agents |
| `run(steps, &evo)` | Run N steps, returning `Vec<SimulationSnapshot>` |
| `agent_at_sink(&agent, &sink, radius)` | Check proximity |

### `SimulationSnapshot`

A recorded state at one simulation tick.

```rust
pub struct SimulationSnapshot {
    pub step: usize,
    pub agent_positions: Vec<(String, f64, f64)>,
    pub field_magnitude: f64,
    pub convergence_count: usize,
}
```

## How It Works

### Intention Vectors

An `IntentionVector` stores direction θ ∈ [0, 2π) and magnitude m ∈ [0, 1]. Vector addition decomposes into Cartesian components, sums them, and reconstructs the polar form:

```
a + b = polar( (a·cos θₐ + b·cos θᵦ)² + (a·sin θₐ + b·sin θᵦ)² )
```

The result magnitude is clamped to 1.0 to keep the field bounded.

### Field Operations

The field uses **central finite differences** for interior points and **forward/backward differences** at boundaries:

- **Gradient**: ∂m/∂x ≈ (m(x+1,y) − m(x−1,y)) / 2
- **Divergence**: ∇·F = ∂(F·cos θ)/∂x + ∂(F·sin θ)/∂y
- **Curl**: ∇×F = ∂(F·sin θ)/∂x − ∂(F·cos θ)/∂y

### Diffusion

Discrete diffusion on a 4-connected grid with rate D and timestep dt:

```
V'(x,y) = (1 − D·dt) · V(x,y) + (D·dt) · avg_neighbors(V)
```

This is a forward-Euler discretization of the heat equation applied to vector fields. The neighbor average uses vector addition (each neighbor scaled by 0.25).

### Decay

Exponential decay of magnitude each timestep:

```
m' = m · (1 − decay_rate · dt)
```

Clamped to [0, 1]. With high decay rates or large dt, this converges to zero.

### Agent Navigation

Agents read the local gradient of field magnitude. If the gradient is nonzero, the agent moves along it (normalized). If the gradient is zero but the local vector is not, the agent follows the local intention direction. Movement is scaled by velocity and sensitivity, and clamped to field bounds.

### Source Radiation

A source at (sx, sy) with intention vector I and radius R contributes to point (px, py):

```
dist = √((px−sx)² + (py−sy)²)
direction = atan2(py−sy, px−sx)
weight = 1 / (1 + dist)
magnitude = I.magnitude × weight   (if dist ≤ R)
```

At the source location itself, the full intention is returned. Beyond the radius, contribution is zero.

## The Math

### Vector Calculus on a Discrete Grid

For a continuous 2D vector field **F** = (Fₓ, Fᵧ), the fundamental operators are:

- **Gradient** of a scalar field φ: ∇φ = (∂φ/∂x, ∂φ/∂y)
- **Divergence**: ∇·**F** = ∂Fₓ/∂x + ∂Fᵧ/∂y — measures net flux. Positive = source, negative = sink.
- **Curl** (2D scalar): ∇×**F** = ∂Fᵧ/∂x − ∂Fₓ/∂y — measures rotation. Zero = irrotational.

This library applies these to the discrete field using finite differences. For a uniform field (all vectors identical), both divergence and curl are exactly zero — the field is divergence-free and curl-free.

### Conservation Law

A field with zero total divergence over its interior is **conserved**: intention is neither created nor destroyed. The `is_conserved(tolerance)` method sums divergence over all interior points and checks if the total is below a tolerance.

### Diffusion as Heat Equation

The diffusion step is a discrete approximation of the **heat equation**:

```
∂V/∂t = D · ∇²V
```

where ∇² is the Laplacian. On a grid, the Laplacian at (x, y) is:

```
∇²V ≈ V(x+1,y) + V(x−1,y) + V(x,y+1) + V(x,y−1) − 4·V(x,y)
```

The implementation blends the current value with the neighbor average proportionally to D·dt, which is stable when D·dt ≤ 0.25.

### Inverse-Distance Weighting

Sources use a 1/(1+d) weighting kernel, a simplified form of **inverse distance weighting** (IDW) interpolation. This ensures:
- Maximum contribution at the source (d = 0 → weight = 1)
- Smooth falloff with distance
- Zero contribution beyond the radius

## Testing

72 tests cover:

- Vector arithmetic (construction, clamping, addition, dot product, alignment)
- Field operations (gradient, divergence, curl on uniform and non-uniform fields)
- Source radiation (inverse-distance, radial direction, boundary behavior)
- Sink saturation model
- Field evolution (diffusion spreading, decay reducing, inject adding)
- Agent navigation (gradient following, stuck detection, bounds clamping)
- Full simulation (multi-step runs, convergence tracking, snapshot recording)
- Serde round-trips for all serializable types

Run with `cargo test`.

## License

MIT
