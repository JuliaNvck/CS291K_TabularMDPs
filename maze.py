import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.collections import PatchCollection
import numpy as np

from mdpsolve_student import *


MAZE_LINE = """
 - - - - 
|s     g|
 - - - - 
"""

MAZE_OPEN = """
 - - - - 
|s      |
         
|       |
         
|       |
         
|      g|
 - - - - 
"""

MAZE_FASTSLOW = """
 - - - - 
|s     g|
   - -   
| |   | |
         
| | | | |
         
|   |   |
 - - - - 
"""

MAZE_FORK = """
 - - - - 
|       |
   - -   
|s|   |g|
   - -   
|       |
 - - - - 
"""

# In this maze with p_success = 1, it's equally good to go right in row 0 or in
# row 1. But with p_success < 1, it's better to use row 0 to avoid getting
# stuck in one of the dead ends.
MAZE_DEADENDS = """
 - - - - - 
|s        |
           
|         |
           
| | | | | |
           
| | | | |g|
 - - - - - 
"""
MAZE_NAMES = ["line", "open", "fastslow", "fork", "deadends"]


ACTIONS = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]]

# for visualizing Q fn
ACTION_TRIS = [
    None,
    np.array([[0, 0], [-1, -1], [1, -1]]),
    np.array([[0, 0], [-1, 1], [1, 1]]),
    np.array([[0, 0], [-1, -1], [-1, 1]]),
    np.array([[0, 0], [1, -1], [1, 1]]),
]


class Maze:
    def __init__(self, maze_str):
        """Constructor.

        Args:
            maze_str (str): Maze represented as a string with linebreaks,
                following the format shown in examples above.
        """
        m = maze_str.strip("\n").split("\n")
        assert len(m) % 2 == 1
        assert len(m[0]) % 2 == 1
        for i in range(1, len(m)):
            assert len(m[i]) == len(m[0])
        self.rows = len(m) // 2
        self.cols = len(m[0]) // 2
        self.m = np.array([list(s) for s in m])


    def draw(self, V=None, Q=None, fig=None, ax=None, vmin=None, vmax=None):
        """Draws the maze, and optionally a value or Q-function, using Matplotlib.

        The (fig, ax) and (vmin, vmax) optional arguments help you make
        animations or interactive step-through code for each iteration of Value
        Iteration. Example usage is shown in main(). The (fig, ax) returned by
        the first draw() call should be passed to subsequent draw() calls.
        Typically vmin = 0, vmax = 1/(1-gamma) should be used; otherwise the
        colorbar range will change between algorithm iterations.

        Args:
            V (array(S)): State value function to draw.
            Q (array(S, A)): State-action value function to draw.
            fig (matplotlib Figure): Figure in which to overwrite.
            ax (matplotlib Axis): Axis in which to overwrite.
            vmin (float): Minimum value for V/Q colorbar. Typically 0.
            vmax (float): Maximum value for V/Q colorbar. Typically 1/(1-gamma).

        Returns:
            fig, ax: Matplotlib Figure and Axis. Identical to (fig, ax) args if
                supplied, otherwise new ones are constructed.
        """
        if V is not None and Q is not None:
            raise ValueError("Can only draw one of V or Q at a time.")
        has_cbar = V is not None or Q is not None
        if (fig is None) != (ax is None):
            raise ValueError("Must supply both fig and ax, or neither.")

        first = fig is None or ax is None
        if first:
            w = self.cols + has_cbar
            h = self.rows
            fig, ax = plt.subplots(constrained_layout=True, figsize=(w, h), subplot_kw=dict(aspect="equal"))
        else:
            ax.clear()

        # room for thick linewidth
        PAD = 0.1
        ax.set(xlim=(-PAD, 2*self.cols+PAD), ylim=(-PAD, 2*self.rows+PAD))
        # so we can use the ascii art coordinate system directly
        if first:
            t = transforms.Affine2D().from_values(0, 1, 1, 0, 0, 0)
            ax.transData = t + ax.transData
        ax.invert_yaxis()
        ax.axis("off")

        # Build patches for either V or Q
        patch = []
        color = []
        label = None

        if V is not None:
            V = V.reshape((self.rows, self.cols))
            for r in range(self.rows):
                for c in range(self.cols):
                    center = np.array([2 * r + 1, 2 * c + 1])
                    patch.append(patches.Rectangle(center - 1, 2, 2))
                    color.append(V[r, c])
            label = "$V^\\star$"

        elif Q is not None:
            arrows = []
            Q = Q.reshape((self.rows, self.cols, -1))
            for r in range(self.rows):
                for c in range(self.cols):
                    # triangles for the moving actions
                    Qbest = np.max(Q[r, c, :])
                    argbest = np.flatnonzero(np.isclose(Q[r, c, :], Qbest))
                    center = np.array([2 * r + 1, 2 * c + 1])
                    for i, tri in list(enumerate(ACTION_TRIS))[1:]:
                        patch.append(patches.Polygon(center + tri, closed=True))
                        color.append(Q[r, c, i])
                        if i in argbest:
                            trimid = np.mean(tri, axis=0)
                            pts = center + 1.2 * trimid - 0.15 * (tri - trimid)
                            arrows.append(patches.Polygon(pts, closed=True))
                    # circle for stay-put action -- on top of tris
                    patch.append(patches.Circle(center, radius=0.5))
                    color.append(Q[r, c, 0])
                    if 0 in argbest:
                        arrows.append(patches.Circle(center, radius=0.1))
            label = "$Q^\\star$"
            pc_arrows = PatchCollection(arrows, facecolors="white", zorder=100)
            ax.add_collection(pc_arrows)

        if V is not None or Q is not None:
            pc = PatchCollection(
                patch,
                cmap="cool",
                edgecolors="black",
                linewidths=0.25,
            )
            pc.set_array(color)
            if vmin is not None or vmax is not None:
                pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
            if first:
                fig.colorbar(pc, ax=ax, label=label, fraction=1/(self.cols+1))

        kwargs = dict(color="black", markersize=15)

        for r in range(self.rows):
            i = 2 * r + 1
            for c in range(self.cols):
                j = 2 * c + 1
                if self.m[i, j] == "s":
                    ax.plot([i], [j], marker="s", **kwargs)
                if self.m[i, j] == "g":
                    ax.plot([i], [j], marker="*", **kwargs)
                # blocked to left, right
                for di in [-1, 1]:
                    lw = 4 if self.m[i + di, j] != " " else 0.25
                    ax.plot([i + di, i + di], [j - 1, j + 1], linewidth=lw, **kwargs)
                # blocked to up, down
                for dj in [-1, 1]:
                    lw = 4 if self.m[i, j + dj] != " " else 0.25
                    ax.plot([i - 1, i + 1], [j + dj, j + dj], linewidth=lw, **kwargs)

        return fig, ax


    def to_mdp(self, p_success=1):
        """Converts the maze into a tabular MDP compatible with your Task 1-3 code.

        Your Bellman / VI code should not need to know how we convert between
        1D state indices and 2D row/column indices in the maze.

        Args:
            p_success: Probability of transitioning to the action's ``desired
                state''. See assignment description for details.

        Returns:
            T, r: Transition dynamics and reward tables. See "Assignment-wide
                conventions" docstring in mdpsolve_student.py for more info.
        """
        S = self.rows * self.cols
        A = 5
        reward = np.zeros((S, A))
        T = np.zeros((S, A, S))
        for r in range(self.rows):
            i = 2 * r + 1
            for c in range(self.cols):
                j = 2 * c + 1
                s = r * self.cols + c
                if self.m[i, j] == "g":
                    reward[s, 0] = 1
                # Compute the next state for all actions first, so we can use
                # them in the randomness.
                nextstate = np.zeros(5, dtype=int)
                for a, (di, dj) in enumerate(ACTIONS):
                    if self.m[i + di, j + dj] not in " sg":
                        # blocked - stay in place
                        nextstate[a] = s
                    else:
                        rr, cc = r + di, c + dj
                        if rr < 0 or rr >= self.rows:
                            raise ValueError("maze is not enclosed")
                        if cc < 0 or cc >= self.cols:
                            raise ValueError("maze is not enclosed")
                        nextstate[a] = rr * self.cols + cc
                for a in range(A):
                    for aa in range(A):
                        p = p_success if aa == a else (1 - p_success) / 4
                        T[s, a, nextstate[aa]] += p

        assert np.allclose(T.sum(axis=-1), 1)
        return T, reward


def main():
    parser = argparse.ArgumentParser(description="Visualize maze and value iteration.")
    parser.add_argument(
        "mode",
        choices=["maze", "V", "Q"],
        help="Output mode: 'maze' for maze visualization, 'V' for value function, 'Q' for Q-function",
    )
    parser.add_argument(
        "--maze",
        choices=MAZE_NAMES,
        default="fastslow",
        help="Example maze to use. See top of maze.py for layouts.",
    )
    parser.add_argument(
        "--psuccess",
        type=float,
        default=1.0,
        help="Probability of transitioning desired state. See assignment for details. (default: 1.0)",
    )
    args = parser.parse_args()

    m = Maze(globals()["MAZE_" + args.maze.upper()])
    T, r = m.to_mdp(p_success=args.psuccess)
    gamma = 0.8
    iters = 25
    Vmax = 1 / (1 - gamma)

    if args.mode == "maze":
        fig, ax = m.draw()
        fig.savefig("maze.pdf")

    elif args.mode == "V":
        V = np.zeros(T.shape[0])
        fig, ax = m.draw(V=V, vmin=0, vmax=Vmax)
        for _ in range(iters):
            m.draw(fig=fig, ax=ax, V=V, vmin=0, vmax=Vmax)
            plt.show(block=False)
            plt.waitforbuttonpress()
            V = bellman_V_opt(T, r, gamma, V)
        fig.savefig("V.pdf")

    elif args.mode == "Q":
        Q = np.zeros(r.shape)
        fig, ax = m.draw(Q=Q, vmin=0, vmax=Vmax)
        for _ in range(iters):
            m.draw(fig=fig, ax=ax, Q=Q, vmin=0, vmax=Vmax)
            plt.show(block=False)
            plt.waitforbuttonpress()
            Q = bellman_Q_opt(T, r, gamma, Q)
        fig.savefig("Q.pdf")


if __name__ == "__main__":
    main()
