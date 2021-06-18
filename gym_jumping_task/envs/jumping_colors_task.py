# coding=utf-8
# MIT License
#
# Copyright 2021 Google LLC
# Copyright (c) 2018 Maluuba Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Jumping Environment with red and green colored obstacles."""

import enum
from gym_jumping_task.envs import jumping_task
import numpy as np

RGB_WHITE = 1.0
OBSTACLE_1 = jumping_task.OBSTACLE_1
OBSTACLE_2 = jumping_task.OBSTACLE_2


class COLORS(enum.Enum):
  RED = 0
  GREEN = 1


class JumpTaskEnvWithColors(jumping_task.JumpTaskEnv):
  """Jumping task with colored obstacle which also affects optimal behavior."""

  def __init__(self, obstacle_color=COLORS.GREEN, **kwargs):
    self._obstacle_color = obstacle_color
    super().__init__(**kwargs, use_colors=True)
    if self._obstacle_color == COLORS.GREEN:
      # Reward provided on colliding with the obstacle when obstacle is green
      self.rewards['collision'] = 100
    else:
      self.rewards['collision'] = 0
    self._already_collided = False

  def _reset(self, *args, **kwargs):
    self._already_collided = False
    return super()._reset(*args, **kwargs)

  def get_state(self):
    """Returns an np array of the screen in RGB."""
    obs = np.zeros((self.scr_h, self.scr_w, 3), dtype=np.float32)

    def _fill_rec(left, up, size, color):
      obs[left: left + size[0], up: up + size[1], :] = color

    def _fill_obstacle(left, up, size):
      right, down = left + size[0], up + size[1]
      for channel in range(3):
        if channel == self._obstacle_color.value:
          obs[left:right, up:down, channel] = 0.5
        else:
          obs[left:right, up:down, channel] = 0.0

    # Add agent and obstacles
    _fill_rec(
        self.agent_pos_x, self.agent_pos_y, self.agent_size, RGB_WHITE)
    if self.two_obstacles:
      # Multiple obstacles
      _fill_obstacle(OBSTACLE_1, self.floor_height, self.obstacle_size)
      _fill_obstacle(OBSTACLE_2, self.floor_height, self.obstacle_size)
    else:
      _fill_obstacle(self.obstacle_position, self.floor_height,
                     self.obstacle_size)

    # Draw the outline of the screen
    obs[0:self.scr_w, 0, :] = RGB_WHITE
    obs[0:self.scr_w, self.scr_h-1, :] = RGB_WHITE
    obs[0, 0:self.scr_h, :] = RGB_WHITE
    obs[self.scr_w-1, 0:self.scr_h, :] = RGB_WHITE

    # Draw the floor
    obs[0:self.scr_w, self.floor_height, :] = RGB_WHITE

    return np.transpose(obs, axes=[1, 0, 2])[::-1]

  def _game_status(self):
    collided, success = super()._game_status()
    if self._obstacle_color == COLORS.GREEN:
      self.done = bool(success)
      collided = (not self._already_collided) and collided
    else:
      self.done = (collided or success)
    self._already_collided = self._already_collided or collided
    return collided, success

  def step(self, action):
    state, reward, done, info = super().step(action)
    if (self.agent_pos_y == self.floor_height) and info['collision']:
      reward += self.rewards['collision']
    return state, reward, done, info

