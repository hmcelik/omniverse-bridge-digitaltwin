# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

import omni.kit.test


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_bridge_configuration_is_current(self):
        from bridge.digitaltwin.bridge_config import (
            NUM_PANELS,
            REAL_BRIDGE_LENGTH,
            REAL_TRUSS_WIDTH,
            REAL_TRUSS_HEIGHT,
            MEMBER_AREA,
            STRAIN_GAUGE_COUNT,
        )

        self.assertEqual(NUM_PANELS, 5)
        self.assertAlmostEqual(REAL_BRIDGE_LENGTH, 0.50)
        self.assertAlmostEqual(REAL_TRUSS_WIDTH, 0.10)
        self.assertAlmostEqual(REAL_TRUSS_HEIGHT, 0.15)
        self.assertGreater(MEMBER_AREA, 0.0)
        self.assertEqual(STRAIN_GAUGE_COUNT, 4)

    async def test_bridge_topology_matches_corrected_warren_truss(self):
        from bridge.digitaltwin.bridge_geometry import _TrussTopology

        topo = _TrussTopology()
        counts = {}
        for _a, _b, member_type in topo.members:
            counts[member_type] = counts.get(member_type, 0) + 1

        self.assertEqual(len(topo.nodes), 22)
        self.assertEqual(len(topo.members), 58)
        self.assertEqual(counts["chord_bottom"], 10)
        self.assertEqual(counts["chord_top"], 8)
        self.assertEqual(counts["diagonal"], 20)
        self.assertEqual(counts["cross"], 5)
        self.assertEqual(counts["deck"], 6)
        self.assertEqual(counts["deck_diagonal"], 9)

        side_diag = next(
            i for i, member in enumerate(topo.members)
            if member[2] == "diagonal")
        deck_diag = next(
            i for i, member in enumerate(topo.members)
            if member[2] == "deck_diagonal")
        self.assertAlmostEqual(topo.member_real_length(side_diag), math.sqrt(0.025))
        self.assertAlmostEqual(topo.member_real_length(deck_diag), math.sqrt(0.02))
        self.assertEqual(topo.diagonal_member_index("L", 2, True), 13)
        self.assertEqual(topo.diagonal_member_index("L", 2, False), 14)
        self.assertEqual(topo.diagonal_member_index("R", 2, True), 32)
        self.assertEqual(topo.diagonal_member_index("R", 2, False), 33)

    async def test_damage_model_records_and_resets_damage(self):
        from bridge.digitaltwin.damage_model import DamageModel

        model = DamageModel(n_members=2)
        model.record_pass_simple({0: 71e6, 1: 0.0})

        self.assertGreater(model.get_damage(0), 0.0)
        self.assertEqual(model.get_damage(1), 0.0)
        self.assertEqual(model.pass_count, 1)

        model.reset()
        self.assertEqual(model.get_damage(0), 0.0)
        self.assertGreater(model.get_crack_size(0), 0.0)
        self.assertEqual(model.get_crack_ratio(0), 0.0)
        self.assertEqual(model.pass_count, 0)

    async def test_sensor_reader_defaults_to_real_websocket_mode(self):
        from bridge.digitaltwin.sensor_reader import (
            ConnectionConfig,
            SensorReader,
            TrafficMode,
        )

        default_config = ConnectionConfig()
        self.assertEqual(default_config.mode, "websocket")
        self.assertEqual(default_config.traffic_mode, TrafficMode.UNIFORM.value)
        self.assertGreater(default_config.traffic_intensity_vpm, 0.0)
        self.assertEqual(len(default_config.gauge_channel_map), 4)

        reader = SensorReader(ConnectionConfig(mode="sim"))
        self.assertEqual(reader.mode, "sim")
        self.assertFalse(reader.is_live)

        reader.set_traffic_mode(TrafficMode.REALISTIC)
        reader.set_traffic_intensity(18.0)
        self.assertEqual(reader.config.traffic_mode, "realistic")
        self.assertEqual(reader.config.traffic_intensity_vpm, 18.0)


