import unittest

import numpy as np
from gym import spaces

from gym_energyplus.envs.energyplus_build_model import build_ep_model
from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp import (
    EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp
)
from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan import (
    EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan
)
from gym_energyplus.envs.energyplus_model_5ZoneAirCooled import (
    EnergyPlusModel5ZoneAirCooled
)


class TestEnergyPlusModel(unittest.TestCase):

    def test_2ZoneDataCenterHVAC_wEconomizer_Temp(self):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(
            model_file="EnergyPlus/Model-22-1-0/2ZoneDataCenterHVAC_wEconomizer_Temp.idf",
            log_dir=None
        )
        self.assertEqual("2ZoneDataCenterHVAC_wEconomizer_Temp", model.model_basename)

        self.assertIsInstance(model.action_space, spaces.Box)
        self.assertEqual((2,), model.action_space.shape)

        self.assertIsInstance(model.observation_space, spaces.Box)
        self.assertEqual((6,), model.observation_space.shape)

        self.assertTupleEqual(
            (22, 1, 0),
            model.energyplus_version
        )
        self.assertEqual(
            "Electricity Demand Rate",
            model.facility_power_output_var_suffix
        )

    def test_2ZoneDataCenterHVAC_wEconomizer_Temp_Eplus_9_3(self):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(
            model_file="EnergyPlus/Model-9-3-0/2ZoneDataCenterHVAC_wEconomizer_Temp.idf",
            log_dir=None
        )
        self.assertEqual("2ZoneDataCenterHVAC_wEconomizer_Temp", model.model_basename)
        self.assertTupleEqual(
            (9, 3, 0),
            model.energyplus_version
        )
        self.assertEqual(
            "Electric Demand Power",
            model.facility_power_output_var_suffix
        )

    def test_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(self):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(
            model_file="EnergyPlus/Model-22-1-0/2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.idf",
            log_dir=None
        )
        self.assertEqual("2ZoneDataCenterHVAC_wEconomizer_Temp_Fan", model.model_basename)

        self.assertIsInstance(model.action_space, spaces.Box)
        self.assertEqual((4,), model.action_space.shape)

        self.assertIsInstance(model.observation_space, spaces.Box)
        self.assertEqual((6,), model.observation_space.shape)

    def test_ep_model_build(self):
        with self.assertRaises(ValueError):
            build_ep_model(model_file="unknown", log_dir=None)

        self.assertIsInstance(
            build_ep_model(
                model_file="EnergyPlus/Model-22-1-0/2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.idf",
                log_dir=None
            ),
            EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan
        )

        self.assertIsInstance(
            build_ep_model(
                model_file="EnergyPlus/5Zone/5ZoneAirCooled.idf",
                log_dir=None
            ),
            EnergyPlusModel5ZoneAirCooled
        )

    def test_5ZoneAirCooled(self):
        model = EnergyPlusModel5ZoneAirCooled(
            model_file="EnergyPlus/5Zone/5ZoneAirCooled.idf",
            log_dir=None
        )
        self.assertEqual("5ZoneAirCooled", model.model_basename)
        self.assertIsInstance(model.action_space, spaces.Box)
        self.assertEqual((10,), model.action_space.shape)
        self.assertIsInstance(model.observation_space, spaces.Box)
        self.assertEqual((16,), model.observation_space.shape)
        self.assertTupleEqual((9, 5, 0), model.energyplus_version)

    def test_5ZoneAirCooled_reward_uses_power(self):
        model = EnergyPlusModel5ZoneAirCooled(
            model_file="EnergyPlus/5Zone/5ZoneAirCooled.idf",
            log_dir=None
        )
        # raw_state = [Tout, Tz1..Tz5, CoolRate1..5, HeatRate1..5]
        raw_state_low_power = np.array([10.0] + [23.5] * 5 + [0.0] * 10, dtype=float)
        raw_state_high_power = np.array([10.0] + [23.5] * 5 + [10000.0] * 10, dtype=float)
        model.set_raw_state(raw_state_low_power)
        r1 = model.compute_reward()
        model.set_raw_state(raw_state_high_power)
        r2 = model.compute_reward()
        self.assertGreater(r1, r2)
