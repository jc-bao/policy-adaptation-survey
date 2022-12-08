from typing import Optional
import gym
import numpy as np
import control as ct
from icecream import ic
import imageio


class CartPoleEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 1.0/60.0  # seconds between state updates
        self.x_threshold = 4.0

        self.A = np.array(
            [[0, 1, 0, 0], 
            [0, 0, -self.masspole/self.masscart*self.gravity, 0], 
            [0, 0, 0, 1], 
            [0, 0, self.gravity/self.length*(1.0+self.masspole/self.masscart), 0]])
        self.B = np.array([[0], [1 / self.masscart], [0], [-1/self.masscart/self.length]])
        self.dynamic_info = self._get_dynamic_info()
        ic(self.dynamic_info)

        self.action_space = gym.spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=float)

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
        x, x_dot, theta, theta_dot = self.state

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        force = action
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole *
                           costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        return np.array(self.state), 0, False, {}

    def reset(self):
        self.state = (0.1, 2.0, 0.0, 0.5)
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

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

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
    env = CartPoleEnv(render_mode="rgb_array")
    state = env.reset()
    vid = []
    # policy1: pole placement
    K = ct.place(env.A, env.B, [-1, -2, -3, -4])
    # policy2: LQR
    Q = np.array([[8, 0, 0, 0],[0, 2, 0, 0],[0, 0, 6, 0],[0, 0, 0, 1]])
    R = 0.001
    K = ct.lqr(env.A, env.B, Q, R)[0]
    for _ in range(120):
        vid.append(env.render())
        act = -K@state
        state, _, _, _ = env.step(act[0])  # take a random action
    imageio.mimsave('../../results/cartpole.gif', vid, fps=30)
    env.close()

if __name__ == "__main__":
    test_cartpole()