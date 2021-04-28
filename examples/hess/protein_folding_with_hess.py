# https://www.sat-x.io/2021/04/28/protein-folding-problem-with-hess/

import sys
import random
import math
import satx.gcc
from gym_lattice.envs import Lattice2DEnv


def oracle(seq):
    global glb, actions
    env.reset()
    reward = 0
    collisions = 0
    info = None
    for i in range(len(seq)):
        if env.done:
            break
        else:
            obs, reward, done, info = env.step(data[seq[i]])
            if done:
                collisions = info['collisions']
                break
    loc = pow(2, -reward) + collisions ** 2
    if loc < glb:
        env.render()
        actions = info['actions']
        print("Episode finished! Reward: {} | Collisions: {}".format(reward, collisions))
        glb = loc
    return loc


if __name__ == '__main__':
    glb = math.inf

    actions = []
    protein = sys.argv[1]
    n = len(protein)
    env = Lattice2DEnv(protein)

    data = n * [0, 1, 2, 3]
    random.shuffle(data)
    seq = satx.hess_sequence(len(data), oracle, fast=False, target=0)
    print(protein)
    print(''.join(actions))
