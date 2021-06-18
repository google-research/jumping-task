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
"""Jumping task."""

from __future__ import print_function

import argparse
import gym
import pygame
from gym import spaces
from gym.utils import seeding
import numpy as np
import time


################## COLORS #####################
# Colors of the different objects on the screen
RGB_WHITE = (255, 255, 255)
RGB_GREY = (128, 128, 128)
RGB_BLACK = (0, 0, 0)
GREYSCALE_WHITE = 1.0
GREYSCALE_GREY = 0.5
###############################################


############### JUMP PARAMETERS ###############
# The jump shape is a `hat`:
# - diagonal going up until the jump height
# - then diagonal going down
JUMP_HEIGHT = 15
JUMP_VERTICAL_SPEED = 1
JUMP_HORIZONTAL_SPEED = 1
###############################################


############ OBSTACLE POSITIONS ###############
# OBSTACLE_*: fixed x positions of two obstacles on the floor.
# Constrained by the shape of the jump.
# This is used as a form of ultimate generalization test.
# Used when two_obstacles is set to True in the environment
OBSTACLE_1 = 20
OBSTACLE_2 = 55
# These are the 6 random positions used in the paper.
ALLOWED_OBSTACLE_X = [20, 30, 40]
ALLOWED_OBSTACLE_Y = [10, 20]
# Max and min positions
LEFT = 14
RIGHT = 48
DOWN = 0
UP = 41
###############################################

class JumpTaskEnv(gym.Env):

  def __init__(self,
               seed=42,
               scr_w=60,
               scr_h=60,
               floor_height=10,
               agent_w=5,
               agent_h=10,
               agent_init_pos=0,
               agent_speed=1,
               obstacle_position=30,
               obstacle_size=(9, 10),
               rendering=False,
               zoom=8,
               slow_motion=False,
               with_left_action=False,
               max_number_of_steps=600,
               two_obstacles=False,
               finish_jump=False,
               use_colors=False):
    """Environment for the jumping task.

    Args:
      scr_w: screen width, by default 60 pixels
      scr_h: screen height, by default 60 pixels
      floor_height: the height of the floor in pixels, by default 10 pixels
      agent_w: agent width, by default 5 pixels
      agent_h: agent height, by default 10 pixels
      agent_init_pos: initial x position of the agent (on the floor), defaults
       to the left of the screen
      agent_speed: agent lateral speed, measured in pixels per time step,
        by default 1 pixel
      obstacle_position: initial x position of the obstacle (on the floor),
        by default 0 pixels, which is the leftmost one
      obstacle_size: width and height of the obstacle, by default (9, 10)
      rendering: display the game screen, by default False
      zoom: zoom applied to the screen when rendering, by default 8
      slow_motion: if True, sleeps for 0.1 seconds at each time step.
        Allows to watch the game at "human" speed when played by the agent, by
        default False
      with_left_action: if True, the left action is allowed, by default False
      max_number_of_steps: the maximum number of steps for an episode, by
        default 600.
      two_obstacles: puts two obstacles on the floor at a given location.
        The ultimate generalization test, by default False
      finish_jump: perform a full jump when the jump action is selected.
        Otherwise an action needs to be selected as usual, by default False.
      use_colors: Whether to use RGB image or not.
    """

    # Initialize seed.
    self.seed(seed)

    self.rewards = {'life': -1, 'exit': 100}
    self.scr_w = scr_w
    self.scr_h = scr_h
    if use_colors:
      self.state_shape = [scr_w, scr_h, 3]
    else:
      self.state_shape = [scr_w, scr_h]

    self.rendering = rendering
    self.zoom = zoom
    if rendering:
      self.screen = pygame.display.set_mode((zoom*scr_w, zoom*scr_h))

    if with_left_action:
      self.legal_actions = [0, 1, 2]
    else:
      self.legal_actions = [0, 1]
    self.nb_actions = len(self.legal_actions)

    self.agent_speed = agent_speed
    self.agent_current_speed = agent_speed * JUMP_HORIZONTAL_SPEED
    self.jumping = [False, None]
    self.agent_init_pos = agent_init_pos
    self.agent_size = [agent_w, agent_h]
    self.obstacle_size = obstacle_size
    self.step_id = 0
    self.slow_motion = slow_motion
    self.max_number_of_steps = max_number_of_steps
    self.finish_jump = finish_jump

    # Min and max positions of the obstacle
    self.min_x_position = LEFT
    self.max_x_position = RIGHT
    self.min_y_position = DOWN
    self.max_y_position = UP

    # Define gym env objects
    self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_shape))
    self.action_space = spaces.Discrete(self.nb_actions)

    self.reset()

  def _game_status(self):
    """Returns two booleans stating whether the agent is touching the obstacle(s) (failure)
    and whether the agent has reached the right end of the screen (success).
    """
    def _overlapping_objects(env, sx, sy):
      return sx + env.obstacle_size[0] > env.agent_pos_x and sx < env.agent_pos_x + env.agent_size[0] \
          and sy + env.obstacle_size[1] > env.agent_pos_y and sy < env.agent_pos_y + env.agent_size[1]

    if self.two_obstacles:
      failure = _overlapping_objects(self, OBSTACLE_1, self.floor_height) or \
          _overlapping_objects(self, OBSTACLE_2, self.floor_height)
    else:
      failure = _overlapping_objects(
          self, self.obstacle_position, self.floor_height)

    success = self.scr_w < self.agent_pos_x + self.agent_size[0]

    self.done = bool(failure or success)

    if self.rendering:
      self.render()
      if self.slow_motion:
        time.sleep(0.1)

    return failure, success

  def _continue_jump(self):
    """Updates the position of the agent while jumping.
    Needs to be called at each discrete step of the jump
    """
    self.agent_pos_x = np.max([self.agent_pos_x + self.agent_current_speed, 0])
    if self.agent_pos_y > self.floor_height + JUMP_HEIGHT:
      self.jumping[1] = "down"
    if self.jumping[1] == "up":
      self.agent_pos_y += self.agent_speed * JUMP_VERTICAL_SPEED
    elif self.jumping[1] == "down":
      self.agent_pos_y -= self.agent_speed * JUMP_VERTICAL_SPEED
      if self.agent_pos_y == self.floor_height:
        self.jumping[0] = False

  def reset(self, ):
    """Resets the game.
    To be called at the beginning of each episode for training as in the paper.
    Sets the obstacle at one of six random positions.
    """
    obstacle_position = self.np_random.choice(ALLOWED_OBSTACLE_X)
    floor_height = self.np_random.choice(ALLOWED_OBSTACLE_Y)
    return self._reset(obstacle_position, floor_height)

  def _reset(self, obstacle_position=30, floor_height=10, two_obstacles=False):
    """Resets the game.
    Allows to set different obstacle positions and floor heights

    Args:
      obstacle_position: the x position of the obstacle for the new game
      floor_height: the floor height for the new game
      two_obstacles: whether to switch to a two obstacles environment
    """
    self.agent_pos_x = self.agent_init_pos
    self.agent_pos_y = floor_height
    self.agent_current_speed = self.agent_speed * JUMP_HORIZONTAL_SPEED
    self.jumping = [False, None]
    self.step_id = 0
    self.done = False
    self.floor_height = floor_height
    self.two_obstacles = two_obstacles
    if two_obstacles:
      return self.get_state()

    if obstacle_position < self.min_x_position or obstacle_position >= self.max_x_position:
      raise ValueError('The obstacle x position needs to be in the range [{}, {}]'.format(self.min_x_position, self.max_x_position))
    if floor_height < self.min_y_position or floor_height >= self.max_y_position:
      raise ValueError('The floor height needs to be in the range [{}, {}]'.format(self.min_y_position, self.max_y_position))
    self.obstacle_position = obstacle_position
    return self.get_state()


  def close(self):
    """Exits the game and closes the rendering.
    """
    self.done = True
    if self.rendering:
      pygame.quit()

  def seed(self, seed=None):
    """Seed used in the random selection of the obstacle position
    """
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def get_state(self):
    """Returns an np array of the screen in greyscale
    """
    obs = np.zeros((self.scr_h, self.scr_w), dtype=np.float32)

    def _fill_rec(left, up, size, color):
      obs[left: left + size[0], up: up + size[1]] = color

    # Add agent and obstacles
    _fill_rec(self.agent_pos_x, self.agent_pos_y, self.agent_size, 1.0)
    if self.two_obstacles:
      # Multiple obstacles
      _fill_rec(OBSTACLE_1, self.floor_height,
                self.obstacle_size, GREYSCALE_GREY)
      _fill_rec(OBSTACLE_2, self.floor_height,
                self.obstacle_size, GREYSCALE_GREY)
    else:
      _fill_rec(self.obstacle_position, self.floor_height,
                self.obstacle_size, GREYSCALE_GREY)

    # Draw the outline of the screen
    obs[0:self.scr_w, 0] = GREYSCALE_WHITE
    obs[0:self.scr_w, self.scr_h-1] = GREYSCALE_WHITE
    obs[0, 0:self.scr_h] = GREYSCALE_WHITE
    obs[self.scr_w-1, 0:self.scr_h] = GREYSCALE_WHITE

    # Draw the floor
    obs[0:self.scr_w, self.floor_height] = GREYSCALE_WHITE

    return obs.T

  def step(self, action):
    """Updates the game state based on the action selected.
    Returns the state as a greyscale numpy array, the reward obtained by the agent
    and a boolean stating whether the next state is terminal.
    The reward is defined as a +1 for each pixel movement to the right.

    Args
      action: the action to be taken by the agent
    """
    reward = -self.agent_pos_x
    if self.step_id > self.max_number_of_steps:
      print('You have reached the maximum number of steps.')
      self.done = True
      return self.get_state(), 0., self.done, {}
    elif action not in self.legal_actions:
      raise ValueError(
          'We did not recognize that action. '
          'It should be an int in {}'.format(self.legal_actions))
    if self.jumping[0]:
      self._continue_jump()
    elif action == 0:  # right
      self.agent_pos_x += self.agent_speed
      self.agent_current_speed = self.agent_speed * JUMP_HORIZONTAL_SPEED
    elif action == 1:  # jump
      self.jumping = [True, "up"]
      self._continue_jump()
    elif action == 2:  # left, can only be taken if self.with_left_action is set to True
      if self.agent_pos_x > 0:
        self.agent_pos_x -= self.agent_speed
        self.agent_current_speed = -self.agent_speed * JUMP_HORIZONTAL_SPEED
      else:
        self.agent_current_speed = 0

    killed, exited = self._game_status()
    if self.finish_jump:
      # Continue jumping until jump is finished
      # Being in the air is marked by self.jumping[0]
      while self.jumping[0] and not killed and not exited:
        self._continue_jump()
        killed, exited = self._game_status()

    reward += self.agent_pos_x
    if killed:
      reward = self.rewards['life']
    elif exited:
      reward += self.rewards['exit']
    self.step_id += 1
    return self.get_state(), reward, self.done, {'collision': killed}

  def render(self):
    """Render the screen game using pygame.
    """

    if not self.rendering:
      return
    pygame.event.pump()
    self.screen.fill(RGB_BLACK)
    pygame.draw.line(self.screen, RGB_WHITE,
                    [0, self.zoom*(self.scr_h-self.floor_height)],
                    [self.zoom*self.scr_w, self.zoom*(self.scr_h-self.floor_height)], 1)
    agent = pygame.Rect(self.zoom*self.agent_pos_x,
                        self.zoom*(self.scr_h-self.agent_pos_y-self.agent_size[1]),
                        self.zoom*self.agent_size[0],
                        self.zoom*self.agent_size[1])
    pygame.draw.rect(self.screen, RGB_WHITE, agent)

    if self.two_obstacles:
      obstacle = pygame.Rect(self.zoom*OBSTACLE_1,
                             self.zoom*(self.scr_h-self.floor_height-self.obstacle_size[1]),
                             self.zoom*self.obstacle_size[0],
                             self.zoom*self.obstacle_size[1])
      pygame.draw.rect(self.screen, RGB_GREY, obstacle)
      obstacle = pygame.Rect(self.zoom*OBSTACLE_2,
                             self.zoom*(self.scr_h-self.floor_height-self.obstacle_size[1]),
                             self.zoom*self.obstacle_size[0],
                             self.zoom*self.obstacle_size[1])
    else:
      obstacle = pygame.Rect(self.zoom*self.obstacle_position,
                             self.zoom*(self.scr_h-self.obstacle_size[1]-self.floor_height),
                             self.zoom*self.obstacle_size[0],
                             self.zoom*self.obstacle_size[1])

    pygame.draw.rect(self.screen, RGB_GREY, obstacle)

    pygame.display.flip()

def test(args):
  env = JumpTaskEnv(scr_w=args.scr_w, scr_h=args.scr_h, floor_height=args.floor_height,
                    agent_w=args.agent_w, agent_h=args.agent_h, agent_init_pos=args.agent_init_pos, agent_speed=args.agent_speed,
                    obstacle_position=args.obstacle_position, obstacle_size=args.obstacle_size,
                    rendering=True, zoom=args.zoom, slow_motion=True, with_left_action=args.with_left_action,
                    max_number_of_steps=args.max_number_of_steps, two_obstacles=args.two_obstacles, finish_jump=args.finish_jump)
  env.render()
  score = 0
  while not env.done:
    action = None
    if env.jumping[0] and env.finish_jump:
      action = 3
    else:
      events = pygame.event.get()
      for event in events:
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_RIGHT:
            action = 0
          elif event.key == pygame.K_UP:
            action = 1
          elif event.key == pygame.K_LEFT and args.with_left_action:
            action = 2
          elif event.key == pygame.K_e:
            env.exit()
          else:
            action = 'unknown'
    if action is None:
      continue
    elif action == 'unknown':
      print('We did not recognize that action. Please use the arrows to move the agent or the \'e\' key to exit.')
      continue
    _, r, term, _ = env.step(action)
    env.render()
    score += r
    print('Agent position: {:2d} | Reward: {:2d} | Terminal: {}'.format(
        env.agent_pos_x, r, term))
  print('---------------')
  print('Final score: {:2d}'.format(int(score)))
  print('---------------')


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="Options to test the environment")
  parser.add_argument('--scr_w', type=int, default=60,
                      help='screen width, by default 60 pixels')
  parser.add_argument('--scr_h', type=int, default=60,
                      help='screen height, by default 60 pixels')
  parser.add_argument('--floor_height', type=int, default=10,
                      help='the y position of the floor in pixels, by default 10 pixels')
  parser.add_argument('--agent_w', type=int, default=5,
                      help='agent width, by default 5 pixels')
  parser.add_argument('--agent_h', type=int, default=10,
                      help='agent height, by default 10 pixels')
  parser.add_argument('--agent_init_pos', type=int, default=0,
                      help='initial x position of the agent(on the floor), defaults to the left of the screen')
  parser.add_argument('--agent_speed', type=int, default=1,
                      help='agent lateral speed, measured in pixels per time step, by default 1 pixel')
  parser.add_argument('--obstacle_position', type=int, default=0,
                      help='initial x position of the obstacle (on the floor), by default 0 pixels, which is the leftmost one')
  parser.add_argument('--obstacle_size', type=int, default=(9,10),
                      help='width and height of the obstacle, by default(9, 10)')
  parser.add_argument('--zoom', type=int, default=8,
                      help='zoom applied to the screen when rendering, by default 8')
  parser.add_argument('--with_left_action', action='store_true',
                      help='flag, if present, the left action is allowed, by default False')
  parser.add_argument('--max_number_of_steps', type=int, default=600,
                      help='the maximum number of steps for an episode, by default 600.')
  parser.add_argument('--two_obstacles', action='store_true', help='flag, if present: puts two obstacles on the floor at a given location. ' +
                      'The ultimate generalization test, by default False')
  parser.add_argument('--finish_jump', action='store_true', help='flag, if present: perform a full jump when the jump action is selected. ' +
                      'Otherwise an action needs to be selected as usual, by default False')
  args = parser.parse_args()

  test(args)
