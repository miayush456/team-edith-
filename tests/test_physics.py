from __future__ import annotations

import unittest

import numpy as np

from racing_event.config import CarConfig
from racing_event.physics import CarState, CarStateBatch, step_car, step_car_batch


class PhysicsTests(unittest.TestCase):
    def test_throttle_accelerates_and_brake_reduces_speed(self) -> None:
        config = CarConfig()
        state = CarState(0.0, 0.0, 0.0)
        for _ in range(20):
            state = step_car(state, np.asarray([0.0, 1.0]), config, 1.0 / 30.0)
        self.assertGreater(state.speed, 0.0)
        before_braking = state.speed
        for _ in range(10):
            state = step_car(state, np.asarray([0.0, -1.0]), config, 1.0 / 30.0)
        self.assertLess(state.speed, before_braking)

    def test_steering_rate_is_limited(self) -> None:
        config = CarConfig()
        state = CarState(0.0, 0.0, 0.0)
        next_state = step_car(state, np.asarray([1.0, 0.0]), config, 1.0 / 30.0)
        self.assertLessEqual(abs(next_state.steering_angle), config.steering_rate_rad_s / 30.0 + 1.0e-12)

    def test_same_state_and_action_are_bitwise_repeatable(self) -> None:
        config = CarConfig()
        action = np.asarray([0.25, 0.7])
        left = step_car(CarState(1.0, 2.0, 0.3, speed=4.0), action, config, 1.0 / 30.0)
        right = step_car(CarState(1.0, 2.0, 0.3, speed=4.0), action, config, 1.0 / 30.0)
        self.assertEqual(left.__dict__, right.__dict__)

    def test_longitudinal_braking_reduces_available_lateral_grip(self) -> None:
        config = CarConfig()
        state = CarState(
            0.0,
            0.0,
            0.0,
            speed=50.0,
            steering_angle=config.max_steering_rad,
        )
        coasting = step_car(state, np.asarray([1.0, 0.0]), config, 1.0 / 30.0)
        braking = step_car(state, np.asarray([1.0, -1.0]), config, 1.0 / 30.0)
        self.assertLess(abs(braking.yaw_rate), abs(coasting.yaw_rate))

    def test_scalar_and_batch_physics_match_with_runoff_grip(self) -> None:
        config = CarConfig()
        scalar_states = [
            CarState(1.0, 2.0, 0.3, speed=24.0, steering_angle=0.2),
            CarState(-3.0, 1.0, -0.4, speed=31.0, steering_angle=-0.15),
        ]
        actions = np.asarray([[0.8, 0.9], [-0.6, -0.8]], dtype=np.float64)
        grips = np.asarray([1.0, 0.55], dtype=np.float64)
        batch = CarStateBatch(
            x=np.asarray([state.x for state in scalar_states]),
            y=np.asarray([state.y for state in scalar_states]),
            heading=np.asarray([state.heading for state in scalar_states]),
            speed=np.asarray([state.speed for state in scalar_states]),
            lateral_velocity=np.asarray([state.lateral_velocity for state in scalar_states]),
            yaw_rate=np.asarray([state.yaw_rate for state in scalar_states]),
            steering_angle=np.asarray([state.steering_angle for state in scalar_states]),
            previous_steering_command=np.asarray(
                [state.previous_steering_command for state in scalar_states]
            ),
            previous_longitudinal_command=np.asarray(
                [state.previous_longitudinal_command for state in scalar_states]
            ),
        )
        actual = step_car_batch(batch, actions, config, 1.0 / 30.0, grips)
        expected = [
            step_car(state, action, config, 1.0 / 30.0, float(grip))
            for state, action, grip in zip(scalar_states, actions, grips)
        ]
        field_names = (
            "x",
            "y",
            "heading",
            "speed",
            "lateral_velocity",
            "yaw_rate",
            "steering_angle",
            "previous_steering_command",
            "previous_longitudinal_command",
        )
        for field_name, actual_field in zip(field_names, actual.fields):
            np.testing.assert_array_equal(
                actual_field,
                np.asarray([getattr(state, field_name) for state in expected]),
            )

    def test_nonpositive_dt_and_grip_are_rejected(self) -> None:
        config = CarConfig()
        state = CarState(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            step_car(state, np.zeros(2), config, 0.0)
        with self.assertRaises(ValueError):
            step_car(state, np.zeros(2), config, 1.0 / 30.0, 0.0)


if __name__ == "__main__":
    unittest.main()
