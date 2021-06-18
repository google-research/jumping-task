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

"""Jumping Environment with obstacle and agent coordinates as state."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from gym import spaces
from gym_jumping_task.envs import jumping_task
import numpy as np

LEFT = jumping_task.LEFT
RIGHT = jumping_task.RIGHT


class JumpTaskEnvWithCoordinates(jumping_task.JumpTaskEnv):
  """Jumping Environment with obstacle and agent coordinates as state."""

  def __init__(self, *args, **kwargs):
    super(JumpTaskEnvWithCoordinates, self).__init__(*args, **kwargs)
    low = np.array([1.0 - RIGHT, 0.0])
    high = np.array([56.0 - LEFT, 16.0])

    self.state_shape = (3,)
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

  def get_state(self):
    coordinates = [self.agent_pos_x - self.obstacle_position,
                   self.agent_pos_y - self.floor_height]
    return np.array(coordinates)
