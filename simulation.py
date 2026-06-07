import tomllib

import numpy as np

import lj

class Config:
    def __init__(self, path):
        with open(path, 'rb') as f:
            conf = tomllib.load(f)


        self.sigma           = conf['atom']['sigma']
        self.epsilon         = conf['atom']['epsilon']
        self.mass            = conf['atom']['mass']
        self.symbol          = conf['atom']['symbol']
        self.cutoff          = conf['atom']['cutoff']
        self.number_of_atoms = conf['simulation']['number_of_atoms']
        self.steps           = conf['simulation']['steps']
        self.dt              = conf['simulation']['dt']
        self.box_len         = conf['simulation']['box_length']
        self.temperature     = conf['simulation']['temperature']
        self.density         = conf['simulation']['density']


def init_positions(number_of_atoms, box_length):
    points_per_dim = int(np.ceil(number_of_atoms ** (1/3)))

    positions = np.zeros((number_of_atoms, 3))
    spacing = box_length / points_per_dim

    i = 0
    for x in range(points_per_dim):
        for y in range(points_per_dim):
            for z in range(points_per_dim):
                positions[i] = [
                    (x * spacing) + (spacing / 2.0),
                    (y * spacing) + (spacing / 2.0),
                    (z * spacing) + (spacing / 2.0)
                ]
                i += 1
        
    # make sure center of mass is the same as center of our geometric box
    offset = np.mean(positions, axis=0) - (box_length / 2.0)
    positions -= offset
    return positions


def init_velocities(number_of_atoms, temperature, seed=42):
    rng = np.random.default_rng(seed)
    velocities = rng.normal(0.0, np.sqrt(temperature), (number_of_atoms, 3))

    # this removes the COM motion
    velocities -= np.mean(velocities, axis=0)

    # remove 3 degree of freedom because we removed the COM motion
    degrees_of_freedom = 3.0 * (number_of_atoms - 1.0)
    scaler = degrees_of_freedom * temperature / np.sum(velocities ** 2)
    velocities *= scaler
    return velocities


def kinetic(velocities, mass=1.0):
    return 0.5 * mass * np.sum(velocities ** 2)


def temperature(kinetic, number_of_atoms):
    degrees_of_freedom = 3.0 * (number_of_atoms - 1.0)
    return 2 * kinetic / degrees_of_freedom * boltzmann_const


def entropy(velocities, bins=20):
    speeds = np.linalg.norm(velocities, axis=1)
    counts, _ = np.histogram(speeds, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0.0]
    return -1.0 * np.sum(probs * np.log(probs))


class Simulation:
    def __init__(self, config):
        self.conf = config

    def setup(self):
        self.pos = init_positions(self.conf.number_of_atoms, self.conf.box_len)
        self.vel = init_velocities(self.conf.number_of_atoms, self.conf.temperature)
        self.acc, _ = lj.compute(self.pos, self.conf.sigma, self.conf.epsilon, self.conf.cutoff, self.conf.box_len)

    def state(self):
        pass

    def run(self):
        pass
