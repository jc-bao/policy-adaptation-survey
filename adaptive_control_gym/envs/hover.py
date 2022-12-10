from typing import Optional
import gym
import numpy as np
import control as ct
from icecream import ic
import imageio

from adaptive_control_gym import controllers as ctrl


class HoverEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.screen_width = 50
        self.screen_height = 200
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.mass = 1.0
        self.tau = 1.0/60.0  # seconds between state updates
        self.x_threshold = 5.0
        self.max_force = 10.0

        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1 / self.mass]])
        self.dynamic_info = self._get_dynamic_info()
        ic(self.dynamic_info)

        self.action_space = gym.spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=float)

    def _get_dynamic_info(self):
        eig_A = np.linalg.eig(self.A)
        ctrb_AB = ct.ctrb(self.A, self.B)
        ctrb_rank = np.linalg.matrix_rank(ctrb_AB)
        return {
            'eig_A': eig_A,
            'ctrb_AB': ctrb_AB,
            'ctrb_rank': ctrb_rank,
        }
    
    def step(self, action):
        action = np.clip(action, -self.max_force, self.max_force)
        x, x_dot = self.state
        self.force = action
        xacc = self.force / self.mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc

        self.state = (x, x_dot)

        return np.array(self.state), 0, False, {}

    def reset(self):
        self.state = (4.0, 4.0)
        self.force = 0.0
        return self.state

    def render(self, mode='human'):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_height = self.x_threshold * 2
        scale = self.screen_height / world_height
        cartwidth = 15.0
        cartheight = 3.0

        if self.state is None:
            return None
        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        carty = x[0] * scale + self.screen_height / 2.0  # MIDDLE OF CART
        cartx = self.screen_width//2  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        # force line
        gfxdraw.vline(self.surf, self.screen_width//2, int(carty), int(carty+self.force), (1, 1, 0)) 

        gfxdraw.hline(self.surf, 0, self.screen_width, self.screen_height//2, (1, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(60)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

def test_cartpole():
    np.set_printoptions(precision=3, suppress=True)
    env = HoverEnv(render_mode="rgb_array")
    state = env.reset()
    vid = []
    # policy1: pole placement
    K = ct.place(env.A, env.B, [-1, -2])
    # policy2: LQR
    Q = np.array([[50, 0],[0, 1]])
    R = 1
    policy = ctrl.LRQ(env.A, env.B, Q, R)
    for _ in range(180):
        vid.append(env.render())
        act = policy.select_action(state)
        state, _, _, _ = env.step(act)  # take a random action
    imageio.mimsave('../../results/hover.gif', vid, fps=30)
    env.close()

if __name__ == "__main__":
    test_cartpole()