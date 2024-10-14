use nalgebra::{Rotation2, Vector2};

// we set c = 1 for simplicity
// but we preserve the constant in the formulas
// for clarity :)

// c = speed of light
pub const C: f32 = 1.0;
// c^2
pub const C2: f32 = C * C;

// some additional unit definitions:
//  cf = c * frame length = lightframe
//  ch = c * simulation step = lightstep
// these should probably be the same thing in general
// but i'm separating them for clarity

// lorentz transformations with transverse velocities :P
// so we rotate other to [v, 0] (applying the same transformation to self)
// then apply the transverse velocity formula
// then rotate back
pub fn velocity_addition_2d(velocity: Vector2<f32>, other: Vector2<f32>) -> Vector2<f32> {
    let theta = if other.x == 0.0 {
        std::f32::consts::FRAC_PI_4 * -other.y.signum()
    } else {
        -(other.y / other.x).atan()
    };
    let rot = Rotation2::new(theta);
    let v = rot * other;
    assert_eq!(v.y, 0.0); // TODO remove this assert once this is tested
    let v = v.x;
    let gamma_v = gamma(v);
    let u = rot * velocity;
    // thanks for the formula Dr. V :)
    let nx = (u.x + v) / (1.0 + (u.x * v / C2));
    let ny = (u.y / gamma_v) / (1.0 + (u.x * v / C2));
    let n = rot.inverse() * Vector2::new(nx, ny);
    n
}

pub fn gamma(v: f32) -> f32 {
    (1.0 - v * v / C2).sqrt().recip()
}
