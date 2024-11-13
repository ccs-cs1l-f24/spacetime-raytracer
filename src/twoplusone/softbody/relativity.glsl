const float C2 = 1.0;

// at least we don't need velocity addition for this lol

float gamma(float v) {
    return inversesqrt(1.0 - v * v / C2);
}

// relativistic mass
float r_mass(Particle p) {
    return gamma(length(p.ground_vel)) * p.rest_mass;
}

// relativistic momentum
vec2 r_momentum(Particle p) {
    return r_mass(p) * p.ground_vel;
}

// relativistic energy
float r_energy(Particle p) {
    return r_mass(p) * C2;
}

// relativistic kinetic energy
float r_ke(Particle p) {
    return r_energy(p) - p.rest_mass * C2;
}

vec2 nr_acc(vec2 F, vec2 v, float m0) {
    return F / m0;
}

// https://en.wikipedia.org/wiki/Relativistic_mechanics#Force
vec2 r_acc(vec2 F, vec2 v, float m0) {
    return 1.0/(m0 * gamma(length(v))) * (F - dot(v, F)*v/C2);
}
