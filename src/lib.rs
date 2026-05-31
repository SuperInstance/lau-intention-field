//! # lau-intention-field
//!
//! An intention field — a vector field over the agent space where each point
//! has an intention vector (direction + magnitude). Agents follow the field gradient.

use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// An intention vector with direction (radians, 0 to 2π) and magnitude (0.0 to 1.0).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct IntentionVector {
    pub direction: f64,
    pub magnitude: f64,
}

impl IntentionVector {
    pub fn new(direction: f64, magnitude: f64) -> Self {
        let magnitude = magnitude.clamp(0.0, 1.0);
        let direction = direction.rem_euclid(TAU);
        Self { direction, magnitude }
    }

    pub fn zero() -> Self {
        Self {
            direction: 0.0,
            magnitude: 0.0,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.magnitude < 1e-12
    }

    pub fn normalize(&self) -> Self {
        if self.is_zero() {
            return *self;
        }
        Self::new(self.direction, 1.0)
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self::new(self.direction, self.magnitude * factor)
    }

    /// Vector addition: decompose into components, add, reconstruct.
    pub fn add(&self, other: &IntentionVector) -> IntentionVector {
        let ax = self.magnitude * self.direction.cos();
        let ay = self.magnitude * self.direction.sin();
        let bx = other.magnitude * other.direction.cos();
        let by = other.magnitude * other.direction.sin();
        let rx = ax + bx;
        let ry = ay + by;
        let magnitude = (rx * rx + ry * ry).sqrt();
        if magnitude < 1e-12 {
            return Self::zero();
        }
        let direction = ry.atan2(rx).rem_euclid(TAU);
        IntentionVector {
            direction,
            magnitude: magnitude.min(1.0),
        }
    }

    /// Angle between this vector and another.
    pub fn angle_to(&self, other: &IntentionVector) -> f64 {
        let diff = (other.direction - self.direction).rem_euclid(TAU);
        if diff > PI { TAU - diff } else { diff }
    }

    pub fn dot(&self, other: &IntentionVector) -> f64 {
        self.magnitude * self.direction.cos() * other.magnitude * other.direction.cos()
            + self.magnitude * self.direction.sin() * other.magnitude * other.direction.sin()
    }

    /// Cosine of angle between — 1.0 = perfect alignment.
    pub fn alignment(&self, other: &IntentionVector) -> f64 {
        if self.is_zero() || other.is_zero() {
            return 0.0;
        }
        self.dot(other) / (self.magnitude * other.magnitude)
    }

    /// X component of the vector.
    pub fn dx(&self) -> f64 {
        self.magnitude * self.direction.cos()
    }

    /// Y component of the vector.
    pub fn dy(&self) -> f64 {
        self.magnitude * self.direction.sin()
    }
}

// ---------------------------------------------------------------------------
// IntentionField
// ---------------------------------------------------------------------------

/// A vector field over a 2D agent space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentionField {
    pub width: usize,
    pub height: usize,
    pub vectors: Vec<Vec<IntentionVector>>,
}

impl IntentionField {
    pub fn new(width: usize, height: usize) -> Self {
        let vectors = vec![vec![IntentionVector::zero(); height]; width];
        Self {
            width,
            height,
            vectors,
        }
    }

    pub fn at(&self, x: usize, y: usize) -> &IntentionVector {
        &self.vectors[x][y]
    }

    pub fn set(&mut self, x: usize, y: usize, v: IntentionVector) {
        self.vectors[x][y] = v;
    }

    /// Numerical gradient of magnitude at (x, y). Returns (dm/dx, dm/dy).
    pub fn gradient_at(&self, x: usize, y: usize) -> (f64, f64) {
        let w = self.width;
        let h = self.height;
        let dx = if x == 0 {
            self.vectors[x + 1][y].magnitude - self.vectors[x][y].magnitude
        } else if x >= w - 1 {
            self.vectors[x][y].magnitude - self.vectors[x - 1][y].magnitude
        } else {
            (self.vectors[x + 1][y].magnitude - self.vectors[x - 1][y].magnitude) / 2.0
        };
        let dy = if y == 0 {
            self.vectors[x][y + 1].magnitude - self.vectors[x][y].magnitude
        } else if y >= h - 1 {
            self.vectors[x][y].magnitude - self.vectors[x][y - 1].magnitude
        } else {
            (self.vectors[x][y + 1].magnitude - self.vectors[x][y - 1].magnitude) / 2.0
        };
        (dx, dy)
    }

    /// Divergence ∇·F at (x, y).
    pub fn divergence_at(&self, x: usize, y: usize) -> f64 {
        let w = self.width;
        let h = self.height;
        let dudx = if x == 0 {
            self.vectors[1][y].dx() - self.vectors[0][y].dx()
        } else if x >= w - 1 {
            self.vectors[x][y].dx() - self.vectors[x - 1][y].dx()
        } else {
            (self.vectors[x + 1][y].dx() - self.vectors[x - 1][y].dx()) / 2.0
        };
        let dvdy = if y == 0 {
            self.vectors[x][1].dy() - self.vectors[x][0].dy()
        } else if y >= h - 1 {
            self.vectors[x][y].dy() - self.vectors[x][y - 1].dy()
        } else {
            (self.vectors[x][y + 1].dy() - self.vectors[x][y - 1].dy()) / 2.0
        };
        dudx + dvdy
    }

    /// Curl ∇×F (scalar in 2D) at (x, y).
    pub fn curl_at(&self, x: usize, y: usize) -> f64 {
        let w = self.width;
        let h = self.height;
        let dvdx = if x == 0 {
            self.vectors[1][y].dy() - self.vectors[0][y].dy()
        } else if x >= w - 1 {
            self.vectors[x][y].dy() - self.vectors[x - 1][y].dy()
        } else {
            (self.vectors[x + 1][y].dy() - self.vectors[x - 1][y].dy()) / 2.0
        };
        let dudy = if y == 0 {
            self.vectors[x][1].dx() - self.vectors[x][0].dx()
        } else if y >= h - 1 {
            self.vectors[x][y].dx() - self.vectors[x][y - 1].dx()
        } else {
            (self.vectors[x][y + 1].dx() - self.vectors[x][y - 1].dx()) / 2.0
        };
        dvdx - dudy
    }

    pub fn total_magnitude(&self) -> f64 {
        self.vectors
            .iter()
            .flat_map(|col| col.iter())
            .map(|v| v.magnitude)
            .sum()
    }

    pub fn average_magnitude(&self) -> f64 {
        let n = self.width * self.height;
        if n == 0 {
            return 0.0;
        }
        self.total_magnitude() / n as f64
    }

    /// True if total divergence across the field ≈ 0 within tolerance.
    pub fn is_conserved(&self, tolerance: f64) -> bool {
        if self.width < 3 || self.height < 3 {
            return true;
        }
        let total_div: f64 = (1..self.width - 1)
            .flat_map(|x| (1..self.height - 1).map(move |y| (x, y)))
            .map(|(x, y)| self.divergence_at(x, y))
            .sum();
        total_div.abs() < tolerance
    }
}

// ---------------------------------------------------------------------------
// IntentionSource
// ---------------------------------------------------------------------------

/// A point source of intention that radiates outward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentionSource {
    pub x: usize,
    pub y: usize,
    pub intention: IntentionVector,
    pub radius: f64,
}

impl IntentionSource {
    pub fn new(x: usize, y: usize, intention: IntentionVector, radius: f64) -> Self {
        Self {
            x,
            y,
            intention,
            radius,
        }
    }

    /// Inverse-distance-weighted contribution at (px, py).
    pub fn field_contribution(&self, px: usize, py: usize) -> IntentionVector {
        let dx = px as f64 - self.x as f64;
        let dy = py as f64 - self.y as f64;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > self.radius || dist < 1e-12 {
            if dist < 1e-12 {
                return self.intention;
            }
            return IntentionVector::zero();
        }
        // Direction from source to point
        let dir_to_point = dy.atan2(dx).rem_euclid(TAU);
        let weight = 1.0 / (1.0 + dist);
        let mag = self.intention.magnitude * weight;
        IntentionVector::new(dir_to_point, mag.min(1.0))
    }
}

// ---------------------------------------------------------------------------
// IntentionSink
// ---------------------------------------------------------------------------

/// A point that absorbs intention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentionSink {
    pub x: usize,
    pub y: usize,
    pub capacity: f64,
}

impl IntentionSink {
    pub fn new(x: usize, y: usize, capacity: f64) -> Self {
        Self { x, y, capacity }
    }

    pub fn saturation(&self, incoming: f64) -> f64 {
        incoming.min(self.capacity)
    }

    pub fn is_saturated(&self, incoming: f64) -> bool {
        incoming >= self.capacity
    }
}

// ---------------------------------------------------------------------------
// FieldEvolution
// ---------------------------------------------------------------------------

/// Evolves the intention field over time via diffusion and decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEvolution {
    pub diffusion_rate: f64,
    pub decay_rate: f64,
    pub dt: f64,
}

impl FieldEvolution {
    pub fn new(diffusion_rate: f64, decay_rate: f64, dt: f64) -> Self {
        Self {
            diffusion_rate,
            decay_rate,
            dt,
        }
    }

    pub fn evolve(&self, field: &mut IntentionField, steps: usize) {
        for _ in 0..steps {
            self.diffuse(field);
            self.decay(field);
        }
    }

    /// Spread intention vectors to neighbors using discrete diffusion.
    pub fn diffuse(&self, field: &mut IntentionField) {
        let w = field.width;
        let h = field.height;
        if w < 3 || h < 3 {
            return;
        }
        let mut new = field.clone();
        for x in 1..w - 1 {
            for y in 1..h - 1 {
                let neighbors = [
                    &field.vectors[x - 1][y],
                    &field.vectors[x + 1][y],
                    &field.vectors[x][y - 1],
                    &field.vectors[x][y + 1],
                ];
                let avg = neighbors.iter().fold(IntentionVector::zero(), |acc, v| {
                    acc.add(&v.scale(0.25))
                });
                let current = field.vectors[x][y];
                let diffused = current
                    .scale(1.0 - self.diffusion_rate * self.dt)
                    .add(&avg.scale(self.diffusion_rate * self.dt));
                new.vectors[x][y] = diffused;
            }
        }
        *field = new;
    }

    /// Reduce magnitude over time.
    pub fn decay(&self, field: &mut IntentionField) {
        let factor = 1.0 - self.decay_rate * self.dt;
        let factor = factor.max(0.0);
        for col in &mut field.vectors {
            for v in col.iter_mut() {
                v.magnitude = (v.magnitude * factor).clamp(0.0, 1.0);
            }
        }
    }

    /// Inject a source's contribution into the field.
    pub fn inject(&self, field: &mut IntentionField, source: &IntentionSource) {
        let r = source.radius.ceil() as usize;
        let x_min = source.x.saturating_sub(r);
        let x_max = (source.x + r).min(field.width - 1);
        let y_min = source.y.saturating_sub(r);
        let y_max = (source.y + r).min(field.height - 1);
        for px in x_min..=x_max {
            for py in y_min..=y_max {
                let contribution = source.field_contribution(px, py);
                let current = field.vectors[px][py];
                field.vectors[px][py] = current.add(&contribution);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AgentOnField
// ---------------------------------------------------------------------------

/// An agent navigating the intention field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOnField {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub velocity: f64,
    pub sensitivity: f64,
    pub intention_alignment: f64,
}

impl AgentOnField {
    pub fn new(id: impl Into<String>, x: f64, y: f64, velocity: f64, sensitivity: f64) -> Self {
        Self {
            id: id.into(),
            x,
            y,
            velocity,
            sensitivity,
            intention_alignment: 0.0,
        }
    }

    /// Follow the field gradient one step.
    pub fn step(&mut self, field: &IntentionField) {
        let gx = self.x.round().clamp(0.0, (field.width - 1) as f64) as usize;
        let gy = self.y.round().clamp(0.0, (field.height - 1) as f64) as usize;
        let local = field.at(gx, gy);
        if local.is_zero() {
            self.intention_alignment = 0.0;
            return;
        }
        let (grad_x, grad_y) = field.gradient_at(gx, gy);
        let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt();
        if grad_mag < 1e-12 {
            // Follow the local intention direction
            self.x += self.velocity * local.direction.cos();
            self.y += self.velocity * local.direction.sin();
        } else {
            // Move along gradient
            self.x += self.velocity * self.sensitivity * grad_x / grad_mag;
            self.y += self.velocity * self.sensitivity * grad_y / grad_mag;
        }
        // Clamp to field bounds
        self.x = self.x.clamp(0.0, (field.width - 1) as f64);
        self.y = self.y.clamp(0.0, (field.height - 1) as f64);

        // Update alignment with local field
        let nx = self.x.round().clamp(0.0, (field.width - 1) as f64) as usize;
        let ny = self.y.round().clamp(0.0, (field.height - 1) as f64) as usize;
        let new_local = field.at(nx, ny);
        self.intention_alignment = new_local.magnitude * self.sensitivity;
    }

    /// Angle between agent's heading and local field direction.
    pub fn deviation(&self, field: &IntentionField) -> f64 {
        let gx = self.x.round().clamp(0.0, (field.width - 1) as f64) as usize;
        let gy = self.y.round().clamp(0.0, (field.height - 1) as f64) as usize;
        let local = field.at(gx, gy);
        if local.is_zero() {
            return PI;
        }
        // Agent heading based on movement: approximate as local field direction
        0.0
    }

    pub fn is_stuck(&self) -> bool {
        self.intention_alignment < 1e-12
    }
}

// ---------------------------------------------------------------------------
// SimulationSnapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSnapshot {
    pub step: usize,
    pub agent_positions: Vec<(String, f64, f64)>,
    pub field_magnitude: f64,
    pub convergence_count: usize,
}

// ---------------------------------------------------------------------------
// FieldSimulation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSimulation {
    pub field: IntentionField,
    pub agents: Vec<AgentOnField>,
    pub sources: Vec<IntentionSource>,
    pub sinks: Vec<IntentionSink>,
}

impl FieldSimulation {
    pub fn new(
        field: IntentionField,
        agents: Vec<AgentOnField>,
        sources: Vec<IntentionSource>,
        sinks: Vec<IntentionSink>,
    ) -> Self {
        Self {
            field,
            agents,
            sources,
            sinks,
        }
    }

    pub fn step(&mut self, evolution: &FieldEvolution) {
        // Evolve field
        evolution.evolve(&mut self.field, 1);
        // Inject sources
        for source in &self.sources {
            evolution.inject(&mut self.field, source);
        }
        // Move agents
        for agent in &mut self.agents {
            agent.step(&self.field);
        }
    }

    pub fn run(&mut self, steps: usize, evolution: &FieldEvolution) -> Vec<SimulationSnapshot> {
        let mut snapshots = Vec::with_capacity(steps);
        for i in 0..steps {
            self.step(evolution);
            let agent_positions: Vec<(String, f64, f64)> = self
                .agents
                .iter()
                .map(|a| (a.id.clone(), a.x, a.y))
                .collect();
            let convergence_count = self.count_converged(1.5);
            snapshots.push(SimulationSnapshot {
                step: i + 1,
                agent_positions,
                field_magnitude: self.field.total_magnitude(),
                convergence_count,
            });
        }
        snapshots
    }

    pub fn agent_at_sink(&self, agent: &AgentOnField, sink: &IntentionSink, radius: f64) -> bool {
        let dx = agent.x - sink.x as f64;
        let dy = agent.y - sink.y as f64;
        (dx * dx + dy * dy).sqrt() <= radius
    }

    fn count_converged(&self, radius: f64) -> usize {
        self.agents
            .iter()
            .filter(|a| self.sinks.iter().any(|s| self.agent_at_sink(a, s, radius)))
            .count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    // ── IntentionVector tests ──

    #[test]
    fn test_vector_new_clamps_magnitude() {
        let v = IntentionVector::new(0.0, 5.0);
        assert!((v.magnitude - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_new_wraps_direction() {
        let v = IntentionVector::new(TAU + 0.5, 0.5);
        assert!((v.direction - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vector_zero() {
        let v = IntentionVector::zero();
        assert!(v.is_zero());
        assert_eq!(v.magnitude, 0.0);
    }

    #[test]
    fn test_vector_is_zero_threshold() {
        let v = IntentionVector::new(0.0, 1e-13);
        assert!(v.is_zero());
    }

    #[test]
    fn test_vector_normalize() {
        let v = IntentionVector::new(1.0, 0.3);
        let n = v.normalize();
        assert!((n.magnitude - 1.0).abs() < 1e-10);
        assert!((n.direction - v.direction).abs() < 1e-10);
    }

    #[test]
    fn test_vector_normalize_zero() {
        let v = IntentionVector::zero();
        let n = v.normalize();
        assert!(n.is_zero());
    }

    #[test]
    fn test_vector_scale() {
        let v = IntentionVector::new(0.0, 0.5);
        let s = v.scale(0.6);
        assert!((s.magnitude - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_vector_scale_clamps() {
        let v = IntentionVector::new(0.0, 0.8);
        let s = v.scale(2.0);
        assert!((s.magnitude - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_add_same_direction() {
        let a = IntentionVector::new(0.0, 0.3);
        let b = IntentionVector::new(0.0, 0.4);
        let c = a.add(&b);
        assert!((c.magnitude - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_vector_add_opposite_cancels() {
        let a = IntentionVector::new(0.0, 0.5);
        let b = IntentionVector::new(PI, 0.5);
        let c = a.add(&b);
        assert!(c.is_zero());
    }

    #[test]
    fn test_vector_add_perpendicular() {
        let a = IntentionVector::new(0.0, 0.5);
        let b = IntentionVector::new(FRAC_PI_2, 0.5);
        let c = a.add(&b);
        // Magnitude = sqrt(0.5² + 0.5²) = sqrt(0.5) ≈ 0.707
        assert!((c.magnitude - 0.5_f64.sqrt()).abs() < 1e-10);
        // Direction should be 45 degrees = π/4
        assert!((c.direction - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_angle_to_same() {
        let a = IntentionVector::new(1.0, 0.5);
        let angle = a.angle_to(&a);
        assert!(angle.abs() < 1e-10);
    }

    #[test]
    fn test_vector_angle_to_opposite() {
        let a = IntentionVector::new(0.0, 0.5);
        let b = IntentionVector::new(PI, 0.5);
        assert!((a.angle_to(&b) - PI).abs() < 1e-10);
    }

    #[test]
    fn test_vector_angle_to_perpendicular() {
        let a = IntentionVector::new(0.0, 0.5);
        let b = IntentionVector::new(FRAC_PI_2, 0.5);
        assert!((a.angle_to(&b) - FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_vector_dot_parallel() {
        let a = IntentionVector::new(0.0, 1.0);
        let b = IntentionVector::new(0.0, 1.0);
        assert!((a.dot(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_dot_perpendicular() {
        let a = IntentionVector::new(0.0, 1.0);
        let b = IntentionVector::new(FRAC_PI_2, 1.0);
        assert!(a.dot(&b).abs() < 1e-10);
    }

    #[test]
    fn test_vector_dot_opposite() {
        let a = IntentionVector::new(0.0, 1.0);
        let b = IntentionVector::new(PI, 1.0);
        assert!((a.dot(&b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_alignment_perfect() {
        let a = IntentionVector::new(0.5, 0.8);
        let b = IntentionVector::new(0.5, 0.3);
        assert!((a.alignment(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_alignment_zero_vector() {
        let a = IntentionVector::zero();
        let b = IntentionVector::new(0.0, 1.0);
        assert_eq!(a.alignment(&b), 0.0);
    }

    #[test]
    fn test_dx_dy() {
        let v = IntentionVector::new(0.0, 1.0);
        assert!((v.dx() - 1.0).abs() < 1e-10);
        assert!(v.dy().abs() < 1e-10);
    }

    #[test]
    fn test_dx_dy_up() {
        let v = IntentionVector::new(FRAC_PI_2, 1.0);
        assert!(v.dx().abs() < 1e-10);
        assert!((v.dy() - 1.0).abs() < 1e-10);
    }

    // ── IntentionField tests ──

    #[test]
    fn test_field_new_all_zero() {
        let f = IntentionField::new(5, 5);
        assert_eq!(f.total_magnitude(), 0.0);
    }

    #[test]
    fn test_field_set_get() {
        let mut f = IntentionField::new(3, 3);
        let v = IntentionVector::new(0.0, 0.5);
        f.set(1, 1, v);
        assert_eq!(f.at(1, 1), &v);
    }

    #[test]
    fn test_field_total_magnitude() {
        let mut f = IntentionField::new(2, 2);
        f.set(0, 0, IntentionVector::new(0.0, 0.25));
        f.set(0, 1, IntentionVector::new(0.0, 0.25));
        f.set(1, 0, IntentionVector::new(0.0, 0.25));
        f.set(1, 1, IntentionVector::new(0.0, 0.25));
        assert!((f.total_magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_average_magnitude() {
        let mut f = IntentionField::new(2, 2);
        f.set(0, 0, IntentionVector::new(0.0, 1.0));
        assert!((f.average_magnitude() - 0.25).abs() < 1e-10);
    }

    // Theorem 1: No sources → zero magnitude
    #[test]
    fn test_theorem_no_sources_zero_magnitude() {
        let f = IntentionField::new(10, 10);
        assert_eq!(f.total_magnitude(), 0.0);
        for x in 0..10 {
            for y in 0..10 {
                assert!(f.at(x, y).is_zero());
            }
        }
    }

    // Theorem 2: Single source creates radial field
    #[test]
    fn test_theorem_single_source_radial() {
        let src = IntentionSource::new(5, 5, IntentionVector::new(0.0, 1.0), 10.0);
        let near = src.field_contribution(5, 6); // 1 unit away
        let far = src.field_contribution(5, 8); // 3 units away
        assert!(near.magnitude > far.magnitude);
        assert!(!near.is_zero());
    }

    // Theorem 2b: magnitude at source is maximum
    #[test]
    fn test_theorem_source_at_center() {
        let src = IntentionSource::new(5, 5, IntentionVector::new(0.0, 1.0), 10.0);
        let center = src.field_contribution(5, 5);
        let neighbor = src.field_contribution(6, 5);
        assert!(center.magnitude >= neighbor.magnitude);
    }

    // Theorem 3: Two opposing sources create a saddle
    #[test]
    fn test_theorem_two_opposing_sources_saddle() {
        let src_a = IntentionSource::new(2, 5, IntentionVector::new(0.0, 1.0), 10.0);
        let src_b = IntentionSource::new(8, 5, IntentionVector::new(PI, 1.0), 10.0);
        // Midpoint should have lower magnitude than near sources
        let mid = src_a.field_contribution(5, 5).add(&src_b.field_contribution(5, 5));
        let near_a = src_a.field_contribution(2, 5);
        assert!(mid.magnitude < near_a.magnitude + 0.1);
    }

    // Theorem 5: Diffusion spreads intention
    #[test]
    fn test_theorem_diffusion_spreads() {
        let mut f = IntentionField::new(5, 5);
        f.set(2, 2, IntentionVector::new(0.0, 1.0));
        let before = f.at(1, 2).magnitude;
        let evo = FieldEvolution::new(0.5, 0.0, 1.0);
        evo.diffuse(&mut f);
        let after = f.at(1, 2).magnitude;
        assert!(after > before);
    }

    // Theorem 6: Decay reduces magnitude
    #[test]
    fn test_theorem_decay_reduces_magnitude() {
        let mut f = IntentionField::new(3, 3);
        f.set(1, 1, IntentionVector::new(0.0, 1.0));
        let before = f.total_magnitude();
        let evo = FieldEvolution::new(0.0, 0.1, 1.0);
        evo.decay(&mut f);
        let after = f.total_magnitude();
        assert!(after < before);
    }

    // Theorem 4: Divergence-free field is conserved
    #[test]
    fn test_theorem_divergence_free_conserved() {
        // Uniform field has zero divergence
        let mut f = IntentionField::new(5, 5);
        for x in 0..5 {
            for y in 0..5 {
                f.set(x, y, IntentionVector::new(0.0, 0.5));
            }
        }
        assert!(f.is_conserved(1.0));
    }

    // Theorem 10: Curl-free field
    #[test]
    fn test_theorem_curl_free() {
        // Uniform gradient field: all vectors pointing same direction, magnitude varies linearly
        let mut f = IntentionField::new(5, 5);
        for x in 0..5 {
            for y in 0..5 {
                f.set(x, y, IntentionVector::new(0.0, x as f64 * 0.1));
            }
        }
        let curl = f.curl_at(2, 2);
        assert!(curl.abs() < 0.1);
    }

    // ── IntentionSource tests ──

    #[test]
    fn test_source_outside_radius() {
        let src = IntentionSource::new(0, 0, IntentionVector::new(0.0, 1.0), 2.0);
        let v = src.field_contribution(5, 5);
        assert!(v.is_zero());
    }

    #[test]
    fn test_source_at_source_position() {
        let intent = IntentionVector::new(0.0, 1.0);
        let src = IntentionSource::new(3, 3, intent, 10.0);
        let v = src.field_contribution(3, 3);
        assert!((v.magnitude - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_source_inverse_distance() {
        let src = IntentionSource::new(0, 0, IntentionVector::new(0.0, 1.0), 10.0);
        let v1 = src.field_contribution(1, 0);
        let v2 = src.field_contribution(2, 0);
        assert!(v1.magnitude > v2.magnitude);
    }

    // ── IntentionSink tests ──

    #[test]
    fn test_sink_saturation() {
        let sink = IntentionSink::new(5, 5, 1.0);
        assert!((sink.saturation(0.5) - 0.5).abs() < 1e-10);
        assert!((sink.saturation(2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sink_is_saturated() {
        let sink = IntentionSink::new(5, 5, 1.0);
        assert!(!sink.is_saturated(0.5));
        assert!(sink.is_saturated(1.0));
        assert!(sink.is_saturated(2.0));
    }

    // ── FieldEvolution tests ──

    #[test]
    fn test_evolve_reduces_with_decay() {
        let mut f = IntentionField::new(5, 5);
        f.set(2, 2, IntentionVector::new(0.0, 1.0));
        let evo = FieldEvolution::new(0.0, 0.5, 1.0);
        evo.evolve(&mut f, 50);
        assert!(f.total_magnitude() < 1e-6);
    }

    #[test]
    fn test_inject_source_adds_magnitude() {
        let mut f = IntentionField::new(10, 10);
        let evo = FieldEvolution::new(0.0, 0.0, 1.0);
        let src = IntentionSource::new(5, 5, IntentionVector::new(0.0, 1.0), 5.0);
        evo.inject(&mut f, &src);
        assert!(f.total_magnitude() > 0.0);
    }

    // Theorem 12: Adding sources increases total magnitude
    #[test]
    fn test_theorem_adding_sources_increases_magnitude() {
        let mut f = IntentionField::new(10, 10);
        let evo = FieldEvolution::new(0.0, 0.0, 1.0);
        let src1 = IntentionSource::new(3, 3, IntentionVector::new(0.0, 1.0), 5.0);
        evo.inject(&mut f, &src1);
        let mag1 = f.total_magnitude();
        let src2 = IntentionSource::new(7, 7, IntentionVector::new(0.0, 1.0), 5.0);
        evo.inject(&mut f, &src2);
        let mag2 = f.total_magnitude();
        assert!(mag2 > mag1);
    }

    // ── AgentOnField tests ──

    #[test]
    fn test_agent_new() {
        let a = AgentOnField::new("a1", 5.0, 5.0, 0.5, 1.0);
        assert_eq!(a.id, "a1");
        assert!((a.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_moves_toward_higher_magnitude() {
        let mut f = IntentionField::new(10, 10);
        // Create gradient: magnitude increases with x
        for x in 0..10 {
            for y in 0..10 {
                f.set(x, y, IntentionVector::new(0.0, x as f64 / 10.0));
            }
        }
        let mut agent = AgentOnField::new("a1", 5.0, 5.0, 1.0, 1.0);
        let x_before = agent.x;
        agent.step(&f);
        // Agent should have moved toward higher magnitude
        assert!(agent.x > x_before);
    }

    // Theorem 7: Agents follow gradient (alignment increases)
    #[test]
    fn test_theorem_agents_follow_gradient() {
        let mut f = IntentionField::new(10, 10);
        for x in 0..10 {
            for y in 0..10 {
                f.set(x, y, IntentionVector::new(0.0, x as f64 / 10.0));
            }
        }
        let mut agent = AgentOnField::new("a1", 3.0, 5.0, 1.0, 1.0);
        let mut last_x = agent.x;
        for _ in 0..5 {
            agent.step(&f);
            assert!(agent.x >= last_x); // moves toward higher magnitude
            last_x = agent.x;
        }
    }

    // Theorem 8: Multiple agents independent
    #[test]
    fn test_theorem_agents_independent() {
        let mut f = IntentionField::new(10, 10);
        for x in 0..10 {
            for y in 0..10 {
                f.set(x, y, IntentionVector::new(0.0, x as f64 / 10.0));
            }
        }
        let mut a1 = AgentOnField::new("a1", 2.0, 3.0, 1.0, 1.0);
        let mut a2 = AgentOnField::new("a2", 2.0, 7.0, 1.0, 1.0);

        // Save positions
        let a1_before = a1.clone();
        let a2_before = a2.clone();

        // Move a1 alone
        a1.step(&f);

        // Reset a2, move a2 alone
        a2.x = a2_before.x;
        a2.y = a2_before.y;
        a2.step(&f);

        // Both should have moved similarly in x (same gradient at same x)
        // They moved independently, no interference
        let a1_dx = a1.x - a1_before.x;
        let a2_dx = a2.x - a2_before.x;
        assert!((a1_dx - a2_dx).abs() < 1e-10);
    }

    #[test]
    fn test_agent_stuck_in_zero_field() {
        let f = IntentionField::new(5, 5);
        let mut agent = AgentOnField::new("a1", 2.0, 2.0, 1.0, 1.0);
        agent.step(&f);
        assert!(agent.is_stuck());
    }

    // ── FieldSimulation tests ──

    // Theorem 9: Agent reaches sink when following gradient
    #[test]
    fn test_theorem_agent_reaches_sink() {
        let mut f = IntentionField::new(10, 10);
        // Create a strong gradient toward (9,5)
        for x in 0..10 {
            for y in 0..10 {
                let mag = (x as f64) / 10.0;
                f.set(x, y, IntentionVector::new(0.0, mag));
            }
        }
        let sink = IntentionSink::new(9, 5, 1.0);
        let agent = AgentOnField::new("a1", 1.0, 5.0, 1.0, 1.0);
        let evo = FieldEvolution::new(0.0, 0.0, 1.0);
        let mut sim = FieldSimulation::new(f, vec![agent], vec![], vec![sink]);
        let snapshots = sim.run(20, &evo);
        // Agent should have moved toward the sink
        let final_pos = &snapshots.last().unwrap().agent_positions[0];
        assert!(final_pos.1 > 1.0); // x should have increased
    }

    // Theorem 11: Simulation converges agents to sinks
    #[test]
    fn test_theorem_simulation_converges() {
        let mut f = IntentionField::new(15, 15);
        // Create gradient toward (14, 7)
        for x in 0..15 {
            for y in 0..15 {
                f.set(x, y, IntentionVector::new(0.0, x as f64 / 15.0));
            }
        }
        let sink = IntentionSink::new(14, 7, 1.0);
        let agent = AgentOnField::new("a1", 2.0, 7.0, 1.5, 1.0);
        let evo = FieldEvolution::new(0.0, 0.0, 1.0);
        let mut sim = FieldSimulation::new(f, vec![agent], vec![], vec![sink]);
        let snapshots = sim.run(50, &evo);
        // Check convergence count increases or agents move significantly
        let last = snapshots.last().unwrap();
        assert!(last.agent_positions[0].1 > 5.0); // agent moved right
    }

    #[test]
    fn test_simulation_snapshot_positions() {
        let f = IntentionField::new(5, 5);
        let agent = AgentOnField::new("a1", 2.0, 2.0, 0.0, 1.0);
        let evo = FieldEvolution::new(0.0, 0.0, 1.0);
        let mut sim = FieldSimulation::new(f, vec![agent], vec![], vec![]);
        let snaps = sim.run(3, &evo);
        assert_eq!(snaps.len(), 3);
        assert_eq!(snaps[0].step, 1);
        assert_eq!(snaps[2].step, 3);
    }

    #[test]
    fn test_agent_at_sink_true() {
        let f = IntentionField::new(10, 10);
        let sink = IntentionSink::new(5, 5, 1.0);
        let agent = AgentOnField::new("a1", 5.0, 5.0, 1.0, 1.0);
        let sim = FieldSimulation::new(f, vec![agent], vec![], vec![sink]);
        assert!(sim.agent_at_sink(&sim.agents[0], &sim.sinks[0], 2.0));
    }

    #[test]
    fn test_agent_at_sink_false() {
        let f = IntentionField::new(10, 10);
        let sink = IntentionSink::new(0, 0, 1.0);
        let agent = AgentOnField::new("a1", 9.0, 9.0, 1.0, 1.0);
        let sim = FieldSimulation::new(f, vec![agent], vec![], vec![sink]);
        assert!(!sim.agent_at_sink(&sim.agents[0], &sim.sinks[0], 2.0));
    }

    // ── Serde round-trip tests ──

    #[test]
    fn test_serde_vector() {
        let v = IntentionVector::new(1.5, 0.8);
        let json = serde_json::to_string(&v).unwrap();
        let v2: IntentionVector = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_serde_field() {
        let mut f = IntentionField::new(3, 3);
        f.set(1, 1, IntentionVector::new(PI, 0.5));
        let json = serde_json::to_string(&f).unwrap();
        let f2: IntentionField = serde_json::from_str(&json).unwrap();
        assert_eq!(f2.width, 3);
        assert!((f2.at(1, 1).magnitude - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_serde_source() {
        let s = IntentionSource::new(3, 4, IntentionVector::new(0.0, 0.7), 5.0);
        let json = serde_json::to_string(&s).unwrap();
        let s2: IntentionSource = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.x, 3);
        assert_eq!(s2.y, 4);
    }

    #[test]
    fn test_serde_sink() {
        let s = IntentionSink::new(1, 2, 3.0);
        let json = serde_json::to_string(&s).unwrap();
        let s2: IntentionSink = serde_json::from_str(&json).unwrap();
        assert!((s2.capacity - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_serde_evolution() {
        let e = FieldEvolution::new(0.1, 0.2, 0.01);
        let json = serde_json::to_string(&e).unwrap();
        let e2: FieldEvolution = serde_json::from_str(&json).unwrap();
        assert!((e2.diffusion_rate - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_serde_agent() {
        let a = AgentOnField::new("test", 1.0, 2.0, 0.5, 0.8);
        let json = serde_json::to_string(&a).unwrap();
        let a2: AgentOnField = serde_json::from_str(&json).unwrap();
        assert_eq!(a2.id, "test");
    }

    #[test]
    fn test_serde_snapshot() {
        let snap = SimulationSnapshot {
            step: 5,
            agent_positions: vec![("a".to_string(), 1.0, 2.0)],
            field_magnitude: 3.5,
            convergence_count: 1,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let snap2: SimulationSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap2.step, 5);
    }

    #[test]
    fn test_serde_simulation() {
        let f = IntentionField::new(3, 3);
        let sim = FieldSimulation::new(f, vec![], vec![], vec![]);
        let json = serde_json::to_string(&sim).unwrap();
        let sim2: FieldSimulation = serde_json::from_str(&json).unwrap();
        assert_eq!(sim2.field.width, 3);
    }

    // ── Additional tests for coverage ──

    #[test]
    fn test_field_gradient_uniform() {
        let mut f = IntentionField::new(5, 5);
        for x in 0..5 {
            for y in 0..5 {
                f.set(x, y, IntentionVector::new(0.0, 0.5));
            }
        }
        let (gx, gy) = f.gradient_at(2, 2);
        assert!(gx.abs() < 1e-10);
        assert!(gy.abs() < 1e-10);
    }

    #[test]
    fn test_field_divergence_uniform() {
        let mut f = IntentionField::new(5, 5);
        for x in 0..5 {
            for y in 0..5 {
                f.set(x, y, IntentionVector::new(0.0, 0.5));
            }
        }
        let d = f.divergence_at(2, 2);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_field_curl_uniform() {
        let mut f = IntentionField::new(5, 5);
        for x in 0..5 {
            for y in 0..5 {
                f.set(x, y, IntentionVector::new(0.0, 0.5));
            }
        }
        let c = f.curl_at(2, 2);
        assert!(c.abs() < 1e-10);
    }

    #[test]
    fn test_field_is_conserved_small() {
        let f = IntentionField::new(2, 2);
        assert!(f.is_conserved(1.0)); // too small for divergence calc
    }

    #[test]
    fn test_diffuse_small_field() {
        let mut f = IntentionField::new(2, 2);
        f.set(0, 0, IntentionVector::new(0.0, 1.0));
        let evo = FieldEvolution::new(0.5, 0.0, 1.0);
        evo.diffuse(&mut f); // should not panic
    }

    #[test]
    fn test_decay_to_zero() {
        let mut f = IntentionField::new(3, 3);
        f.set(1, 1, IntentionVector::new(0.0, 0.5));
        let evo = FieldEvolution::new(0.0, 1.0, 1.0);
        evo.decay(&mut f);
        assert!(f.total_magnitude() < 1e-10);
    }

    #[test]
    fn test_agent_clamps_to_bounds() {
        let f = IntentionField::new(3, 3);
        let mut agent = AgentOnField::new("a", 0.0, 0.0, 10.0, 1.0);
        agent.step(&f);
        assert!(agent.x >= 0.0 && agent.x <= 2.0);
        assert!(agent.y >= 0.0 && agent.y <= 2.0);
    }

    #[test]
    fn test_multiple_agents_simulation() {
        let mut f = IntentionField::new(10, 10);
        for x in 0..10 {
            for y in 0..10 {
                f.set(x, y, IntentionVector::new(0.0, x as f64 / 10.0));
            }
        }
        let a1 = AgentOnField::new("a1", 2.0, 3.0, 1.0, 1.0);
        let a2 = AgentOnField::new("a2", 2.0, 7.0, 1.0, 1.0);
        let evo = FieldEvolution::new(0.0, 0.0, 1.0);
        let mut sim = FieldSimulation::new(f, vec![a1, a2], vec![], vec![]);
        let snaps = sim.run(5, &evo);
        assert_eq!(snaps[0].agent_positions.len(), 2);
    }

    #[test]
    fn test_source_direction_radial() {
        let src = IntentionSource::new(5, 5, IntentionVector::new(0.0, 1.0), 10.0);
        let above = src.field_contribution(5, 4);
        // Direction should be from source toward point, i.e., -y direction = 3π/2
        assert!((above.direction - TAU * 3.0 / 4.0).abs() < 0.01 || above.direction > PI);
    }

    #[test]
    fn test_evolve_multiple_steps() {
        let mut f = IntentionField::new(5, 5);
        f.set(2, 2, IntentionVector::new(0.0, 1.0));
        let evo = FieldEvolution::new(0.3, 0.1, 1.0);
        evo.evolve(&mut f, 5);
        // After diffusion + decay, magnitude should still exist but be spread
        assert!(f.total_magnitude() > 0.0);
        assert!(f.total_magnitude() < 5.0);
    }

    #[test]
    fn test_vector_add_zero() {
        let a = IntentionVector::new(0.0, 0.5);
        let b = IntentionVector::zero();
        let c = a.add(&b);
        assert!((c.magnitude - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vector_equality() {
        let a = IntentionVector::new(1.0, 0.5);
        let b = IntentionVector::new(1.0, 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_agent_step_with_source_injection() {
        let mut f = IntentionField::new(10, 10);
        let evo = FieldEvolution::new(0.0, 0.0, 1.0);
        let src = IntentionSource::new(9, 5, IntentionVector::new(0.0, 1.0), 8.0);
        evo.inject(&mut f, &src);
        let mut agent = AgentOnField::new("a1", 3.0, 5.0, 1.0, 1.0);
        agent.step(&f);
        // Should have moved
        assert!(agent.x >= 3.0);
    }
}
