# lau-intention-field

A vector field over agent space where each point has an intention vector — direction and magnitude. Agents follow the field gradient toward their goals. The field isn't static — it evolves as agents move, collide, and resolve.

## The concept in 60 seconds

An **intention field** is like a magnetic field, but for agent intentions. Each agent deposits intention vectors into the field, and the field gradient tells agents where to go next. Key properties:

- **Deposition:** agents add their intentions to the field
- **Diffusion:** intentions spread to nearby points (like heat diffusion)
- **Advection:** the field carries agents along its gradient
- **Resolution:** when agents reach their target, the field depletes locally

This creates emergent behavior — agents following local gradients produce global coordination without any central planner.

## Quick start

```rust
use lau_intention_field::{IntentionField, IntentionVector, Agent};

let mut field = IntentionField::new(100, 100); // 100x100 grid

// Agent deposits an intention at its location
let agent = Agent::new("hermes").at(50, 50);
let intention = IntentionVector::new(0.7, 0.9); // direction, magnitude
field.deposit(&agent, intention);

// Diffuse the field
field.diffuse(0.1);

// Read the gradient at any point
let gradient = field.gradient_at(30, 30);
```

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-intention-field/issues) or PR.
