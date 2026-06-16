#!/usr/bin/env python

import sys

from simulation import Config, Simulation

def main() -> None:
    conf = Config('config.toml')
    simulation = Simulation(conf)
    simulation.setup()
    simulation.simulate(2)
    return None


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit()
