import unittest
from numbers import Integral

from garage_energyplus_old.env import EnergyPlusMonthTaskEnv


class TestGarageTaskSampler(unittest.TestCase):

    def test_sample_tasks_returns_month_indices(self):
        EnergyPlusMonthTaskEnv.configure(
            energyplus_file="/usr/local/energyplus-9.5.0",
            model_file="EnergyPlus/5Zone/5ZoneAirCooled.idf",
            weather_file=(
                "EnergyPlus/Model-9-5-0/WeatherData/"
                "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
            ),
            log_dir="eplog/test-garage-task-sampler",
            seed=123,
        )
        env = EnergyPlusMonthTaskEnv()

        tasks = env.sample_tasks(20)
        print("sampled tasks:", tasks)

        self.assertEqual(len(tasks), 20)
        self.assertTrue(all(isinstance(task, dict) for task in tasks))
        self.assertTrue(all("month" in task and "zone_id" in task for task in tasks))
        self.assertTrue(all(isinstance(task["month"], Integral) for task in tasks))
        self.assertTrue(all(1 <= task["month"] <= 12 for task in tasks))
        self.assertTrue(
            all(task["zone_id"] in EnergyPlusMonthTaskEnv.AGENT_IDS for task in tasks)
        )

        first_block = tasks[:5]
        self.assertEqual(len({t["month"] for t in first_block}), 1)
        self.assertEqual(
            {t["zone_id"] for t in first_block},
            set(EnergyPlusMonthTaskEnv.AGENT_IDS),
        )

    def test_set_task_updates_current_month(self):
        env = EnergyPlusMonthTaskEnv()
        env.set_task({"month": 7, "zone_id": "zone_4"})
        self.assertEqual(env._task_month, 7)
        self.assertEqual(env._task_zone, "zone_4")


if __name__ == "__main__":
    unittest.main()
