import logging
import math
from random import randrange


class TranslationTaskMixin:
    noise_tolerance: float
    reward_function: str

    # overriding method
    def _set_goal(self):
        x1, y1 = self.goal_min
        x2, y2 = self.goal_max

        goal_x = randrange(x1, x2)
        goal_y = randrange(y1, y2)

        return [goal_x, goal_y]

    # overriding method
    def _compute_reward(self, previous_environment_info, current_environment_info):
        match self.reward_function:
            case "delta":
                return self._reward_function_delta(
                    previous_environment_info, current_environment_info
                )
            case "distance":
                return self._reward_function_linear(
                    previous_environment_info, current_environment_info
                )
            case "staged":
                return self._reward_function_staged(
                    previous_environment_info, current_environment_info
                )
            case "touch_staged":
                return self._reward_function_touch_staged(
                    previous_environment_info, current_environment_info
                )

        return self._reward_function_delta(
            previous_environment_info, current_environment_info
        )

    def _reward_function_delta(
        self, previous_environment_info, current_environment_info
    ):
        reward = 0

        target_goal = current_environment_info["goal"]

        object_pose_current = current_environment_info["poses"]["object"]
        object_pose_previous = previous_environment_info["poses"]["object"]

        if object_pose_current is None:
            # If current pose is None, we cannot compute a delta change
            return 0.0, False

        if object_pose_previous is None:
            # If previous pose is None, we cannot compute a delta change
            return 0.0, False

        # This now converts the poses with respect to reference marker
        # Exclude Z for object
        object_current = self._relative_position(object_pose_current)[:-1]
        object_previous = self._relative_position(object_pose_previous)[:-1]

        goal_difference_before = math.dist(target_goal, object_previous)
        goal_difference_after = math.dist(target_goal, object_current)

        delta_change = goal_difference_before - goal_difference_after

        reward = 0
        if goal_difference_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 1.0
        elif abs(delta_change) > self.noise_tolerance:
            reward = delta_change / max(goal_difference_before, 1e-6)
            reward = max(-1.0, min(1.0, reward))  # Clip reward to [-1, 1]

        return round(reward, 2), False

    def _reward_function_linear(
        self, previous_environment_info, current_environment_info
    ):
        target_goal = current_environment_info["goal"]

        object_pose_current = current_environment_info["poses"]["object"]

        # This now converts the poses with respect to reference marker
        # Exclude Z for object
        if object_pose_current is not None:
            object_current = self._relative_position(object_pose_current)[:-1]

            goal_distance_after = math.dist(target_goal, object_current)
        else:
            # Set to a value outside the goal range
            goal_distance_after = self.goal_range + self.noise_tolerance

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        reward = 0
        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 1.0
        else:
            reward = max(
                0.0,
                1.0
                - (goal_distance_after - self.noise_tolerance)
                / (self.goal_range - self.noise_tolerance),
            )

        return round(reward, 2), False

    # TODO these reward functons need to be refactored
    # overriding method
    def _reward_function_staged(
        self, previous_environment_info, current_environment_info
    ):
        self.goal_range = 25
        self.goal_reward = 4  # goal reward minus the potential moving away negativity
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._relative_position(
            previous_environment_info["poses"]["object"]["position"]
        )[:-1]
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]["position"]
        )[:-1]
        logging.debug(
            f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}"
        )

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        # Staged reward system
        # ----------> S1: Hold <---------- #
        if object_current[1] <= self.bottom_line:
            reward = 1

            delta_changes = goal_distance_before - goal_distance_after

            if (
                self.total_moves < 50
            ):  # actually half of it cuz reward getting run twice per step because of render env
                # ----------> S2: Move <---------- #
                if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
                    # S2: No Move
                    reward += -0.5
                else:
                    # S2: Move
                    reward += 1
                    self.total_moves += 1
                    print("total moves: ", self.total_moves)
            else:
                # ----------> S3: Reach <---------- #
                raw_reward = delta_changes / goal_distance_before

                reward += (
                    1 if raw_reward >= 1 else -1 if raw_reward <= -1 else raw_reward
                )

                # S3: Reach Goal
                if goal_distance_after <= self.goal_range:
                    logging.info("----------Reached the Goal!----------")
                    reward += self.goal_reward + 1

                # S3: No move outside of goal range
                if (
                    goal_distance_after >= self.goal_range
                    and -self.noise_tolerance <= delta_changes <= self.noise_tolerance
                ):
                    reward += -0.5
        # S1: Drop
        else:
            reward = -1

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        print(object_current, reward)

        return reward, done

    # overriding method touch_staged
    def _reward_function_touch_staged(
        self, previous_environment_info, current_environment_info
    ):
        self.goal_range = 25
        self.goal_reward = 4  # goal reward minus the potential moving away negativity
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._relative_position(
            previous_environment_info["poses"]["object"]
        )[:-1]
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]
        logging.debug(
            f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}"
        )

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        # Staged reward system
        # ----------> S1: Hold <---------- #
        if object_current[1] <= self.bottom_line:
            reward = 1

            delta_changes = goal_distance_before - goal_distance_after

            if (
                self.total_moves < 50
            ):  # actually half if it cuz reward getting run twice per step because of render env
                # ----------> S2: Move <---------- #
                if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
                    # S2: No Move
                    reward += -0.5
                else:
                    # S2: Move
                    reward += 1
                    self.total_moves += 1
                    print("total moves: ", self.total_moves)
            else:
                # ----------> S3: Reach <---------- #
                raw_reward = delta_changes / goal_distance_before

                reward += (
                    1 if raw_reward >= 1 else -1 if raw_reward <= -1 else raw_reward
                )

                # S3: Reach Goal
                if goal_distance_after <= self.goal_range:
                    logging.info("----------Reached the Goal!----------")
                    reward += 5

                # S3: No move outside of goal range
                if (
                    goal_distance_after >= self.goal_range
                    and -self.noise_tolerance <= delta_changes <= self.noise_tolerance
                ):
                    reward += -0.5
        # S1: Drop
        else:
            reward = -1

        # Touch Sensor Reward
        if self.use_touch:
            left = current_environment_info["touch"][0]
            right = current_environment_info["touch"][1]

            reward += 0.5 if left or right else 0

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        print(object_current, reward)

        return reward, done
