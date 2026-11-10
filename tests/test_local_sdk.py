"""Extra checks with mono's Python; never opens ROS or hardware."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from tomato_agent.adapters.lia_mono import ReadOnlySource


@unittest.skipUnless(
    importlib.util.find_spec("lia_robots") is not None, "optional mono SDK environment"
)
class LocalSDKTests(unittest.TestCase):
    def test_real_definition_keeps_topics_and_never_connects(self):
        root = Path("/home/user/lpy/li-agent-mono")
        if not root.is_dir():
            self.skipTest("local SDK checkout absent")
        with patch(
            "lia_robots.robots.observation.Ros",
            side_effect=AssertionError("must not open ROS"),
        ):
            source = ReadOnlySource(root, episode_id="sdk-test")
            self.assertIsNone(source.definition._ros)
            self.assertIsNone(source.reader.ros)
            self.assertEqual(len(source.reader.joints), 4)
            self.assertEqual(set(source.reader.sensors), {"cam0", "cam1", "cam2"})
            self.assertEqual(
                source.reader.sensors["cam0"].topic, "/cam0/color/image_raw"
            )
            source.close()


if __name__ == "__main__":
    unittest.main()
