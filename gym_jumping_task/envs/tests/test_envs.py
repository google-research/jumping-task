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

"""Tests for jumping task envs."""

import gym
from gym_jumping_task import COLORS
from gym_jumping_task import registry
import numpy as np
import pytest

spec_list = [s for s in registry.all() if
             s.entry_point.startswith('gym_jumping_task.envs')]


@pytest.mark.parametrize('spec', spec_list)
def test_env(spec):
  """Runs a smoketest on each registered env."""
  # Capture warnings
  with pytest.warns(None) as warnings:
    env = spec.make()

  # Check that dtype is explicitly declared for gym.Box spaces
  for warning_msg in warnings:
    assert 'autodetected dtype' not in str(warning_msg.message)

  ob_space = env.observation_space
  act_space = env.action_space
  ob = env.reset()
  assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(
      ob)
  a = act_space.sample()
  observation, reward, done, _ = env.step(a)
  assert ob_space.contains(
      observation), 'Step observation: {!r} not in space'.format(observation)
  assert np.isscalar(reward), '{} is not a scalar for {}'.format(reward, env)
  # isinstance(done, bool) returns False sometimes
  print(type(done))
  print(isinstance(done, bool))
  assert isinstance(done, bool), 'Expected {} to be a boolean'.format(done)  # pylint: disable=unidiomatic-typecheck
  env.close()


def test_random_rollout():
  """Run a longer rollout on jumping environments."""
  for env in [
      gym.make('jumping-task-v0'), gym.make('jumping-coordinates-task-v0'),
      gym.make('jumping-colors-task-v0')
  ]:
    ob = env.reset()
    for _ in range(10):
      assert env.observation_space.contains(ob)
      a = env.action_space.sample()
      assert env.action_space.contains(a)
      (ob, _, done, _) = env.step(a)
      if done:
        break
    env.close()


def test_colored_env():
  """Run a rollout on green colored jumping environment."""
  for env in [
      gym.make('jumping-colors-task-v0', obstacle_color=COLORS.GREEN)
  ]:
    env.reset()
    rewards = 0
    for _ in range(50):
      _, r, done, _ = env.step(0)
      rewards += r
      if done:
        break
    assert rewards >= env.rewards['collision']
    env.close()
