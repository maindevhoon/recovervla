import unittest
import xml.etree.ElementTree as ET

from recovervla.sim.build_scene import add_neck_tab, vessel


class BottleXmlTests(unittest.TestCase):
    def test_neck_tab_moves_grasp_site_to_the_mouth(self):
        world = ET.Element("worldbody")
        body = vessel(world, "bottle", (0, 0, 0.032), 0.023, 0.10)
        mid = float(body.find("site").get("pos").split()[2])
        self.assertAlmostEqual(mid, 0.05)
        add_neck_tab(body, "bottle", 0.10)
        sites = body.findall("site")
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0].get("name"), "bottle_grasp")
        self.assertAlmostEqual(float(sites[0].get("pos").split()[2]), 0.10)
        tab = body.find("geom[@name='bottle_grip_tab']")
        self.assertIsNotNone(tab)
        self.assertEqual(tab.get("type"), "box")
