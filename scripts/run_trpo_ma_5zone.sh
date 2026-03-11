#!/usr/bin/env bash
set -euo pipefail

ENERGYPLUS=/usr/local/energyplus-9.5.0
ENERGYPLUS_MODEL=/home/wlc/rl-testbed-for-energyplus/EnergyPlus/5Zone/5ZoneAirCooled.idf
ENERGYPLUS_WEATHER=/home/wlc/rl-testbed-for-energyplus/EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw

export ENERGYPLUS
export ENERGYPLUS_MODEL
export ENERGYPLUS_WEATHER

python baselines_energyplus/trpo_mpi/run_energyplus.py --env EnergyPlusMA-Single-v0 --num-timesteps 10000000
