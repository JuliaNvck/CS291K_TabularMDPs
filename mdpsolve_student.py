import numpy as np


"""Assignment-wide conventions:

These notations and arguments mean the same thing throughout the assignment.
Notation: S = number of states, A = number of actions.
The following arguments and/or return values always mean:

    P (array(S, A, S)): Transition dynamics: P[s, a, :] is a S-dimensional
        vector representing the probability distribution over the next state s'
        when starting from state s and taking action a. Note this layout makes
        the expressions look slightly different from the lecture notes math:
        The distribution P(.|s,a) is indexed as P[s, a, :] (or just P[s, a]).
        = P(s'|s,a)

    r (array(S, A)): Reward function: r[s, a] is a scalar representing the
        reward of taking action a from state s.

    gamma (float): The infinite-horizon discount factor, within (0, 1).

    pi (array(S, A)): Policy: pi[s, :] is an A-dimensional vector representing
        the probability distribution over the action a taken at state s. Has
        the same math-vs-code difference as P.
        = pi(a|s)

    V (array(S)): State-value function: current value estimate V(s') for every state starting from state s
"""

"""Task 1: Implement all 4 variants of Bellman operator for tabular MDPs.

The test cases will be randomly generated MDPs. However, you will only receive
binary pass/fail information from Gradescope, so you should not use Gradescope
for development. Instead, you should write your own tests using your test-case
MDPs from Task 2, as well as the Maze MDP tools from maze.py. 
"""

# Takes a current estimate of the value function V and updates it to be closer to the true value of a specific policy pi
# (TπV )(s) = E a∼π(s), s′∼P (·|s,a) [r(s, a) + γV (s′)]
def bellman_V_pi(P, r, gamma, pi, V):
    """Performs one policy Bellman update on the state value function candidate V for policy pi."""
    # immediate reward plus the discounted expected value of the next state s'
    # average over actions a according to policy pi and over next states s' according to transition dynamics P
    expected_values_s_a = ((gamma * (P @ V)) + r)  # shape (S, A)
    weighted_values = pi * expected_values_s_a  # shape (S, A)
    V = np.sum(weighted_values, axis=1)  # shape (S,) - sum over actions to get expected value for each state
    return V


def bellman_Q_pi(P, r, gamma, pi, Q):
    """Performs one policy Bellman update on the state-action value function candidate Q for policy pi."""
    return Q  # TODO: Implement.


def bellman_V_opt(P, r, gamma, V):
    """Performs one optimal Bellman update on the state value function candidate V."""
    return V  # TODO: Implement.


def bellman_Q_opt(P, r, gamma, Q):
    """Performs one optimal Bellman update on the state-action value function candidate Q."""
    return Q  # TODO: Implement.


"""Task 2: Implement some test MDPs.

For the very simplest test MDPs, you can figure out the optimal value function
manually without the use of an algorithm. Write code to construct them in the
_Vopt functions. Your _Vopt function implementations should directly construct
an array full of hand-computed constants; they should not call your bellman or
value_iteration functions.
"""

def mdp_uniform(S, A):
    """An MDP where actions have no effect.

    Desired output MDP: No matter which action is taken, the reward is 1 and
    the next state is uniformly distributed over the entire state set.

    Args:
        S (int): Number of states.
        A (int): Number of actions.

    Returns: a tuple (P, r), as defined in the assignment-wide conventions.
    """
    return np.zeros((S, A, S)), np.zeros((S, A))  #TODO: Implement.


def mdp_uniform_Vopt(S, A, gamma):
    """Returns the optimal state-value function for mdp_uniform.

    Args:
        S (int): Number of states.
        A (int): Number of actions.
        gamma (float): Discount factor.

    Returns: V (array(S)): optimal value function table.
    """
    return np.zeros(S)  #TODO: Implement.


def mdp_statepick(S):
    """An MDP where the action directly selects the next state.

    Desired output MDP: The number of actions is equal to S. When the i'th
        action is taken, the i'th state is the next state with probability 1.
        The reward is 1 for taking action 0 in state 0; zero everywhere else.

    Args:
        S (int): Number of states, is also number of actions.

    Returns: a tuple (P, r), as defined in the assignment-wide conventions.
    """
    return np.zeros((S, S, S)), np.zeros((S, S))  #TODO: Implement.


def mdp_statepick_Vopt(S, gamma):
    """Returns the optimal state-value function for mdp_statepick.

    Args:
        S (int): Number of states.
        gamma (float): Discount factor.

    Returns: V (array(S)): optimal value function table.
    """
    return np.zeros(S)  #TODO: Implement.


def mdp_line(S):
    """An MDP where state are connected in a line graph.

    Desired output MDP: States are ordered from left to right numerically.
    There are three actions: 0 = Go left, 1 = don't move, 2 = go right.
    FOR STATES IN THE MIDDLE OF THE LINE:
        The desired outcome happens with probability 0.8. Each of the other two
        possible outcomes happens with probability 0.1.
    FOR STATES AT EITHER END OF THE LINE:
        The probability for "impossible" outcomes (going left from state 0 or
        right from state S - 1) is reassigned to the nearest valid state. For
        example, if we are in state 0 and take action "left", then the next
        state distribtion is:

            P(s' = 0 | s = 0, a = 0) = 0.9, P(s' = 1 | s = 0, a = 0) = 0.1.
    Reward is 1 for *any* action in state 0, and zero otherwise.

    Args:
        S (int): Number of states. Number of actions is fixed at 3.

    Returns: a tuple (P, r), as defined in the assignment-wide conventions.
    """
    return np.zeros((S, 3, S)), np.zeros((S, 3))  #TODO: Implement.


"""Task 3: Implement all variants of (Q-)Value Iteration and Policy Evaluation.

Use your Bellman operator implementations from Task 1.
Arguments P, r, gamma, and pi are interpreted as in Task 1.
Argument `iters` controls the number of iterations to perform.
NOTE: The initial guess should always be zero, otherwise tests will fail.
"""

def value_iteration_V(P, r, gamma, iters):
    """Performs `iters` steps of value iteration."""
    return np.zeros((r.shape[0]))  # TODO: Implement.


def policy_evaluation_V(P, r, gamma, pi, iters):
    """Performs `iters` steps of policy evaluation for policy pi."""
    return np.zeros((r.shape[0]))  # TODO: Implement.


def value_iteration_Q(P, r, gamma, iters):
    """Performs `iters` steps of Q-value iteration."""
    return np.zeros(r.shape)  # TODO: Implement.


def policy_evaluation_Q(P, r, gamma, pi, iters):
    """Performs `iters` steps of Q- policy evaluation for policy pi."""
    return np.zeros(r.shape)  # TODO: Implement.


"""Task 4: Implement Policy Iteration.

The Policy Iteration code should use your policy_evaluation_Q as a subroutine.
For each "policy evaluation" setup, run it for a fixed amount of iterations,
following the argument `PE_iters`. The initial guess should take action 0 from
all states.

PI should terminate whenever the policy does not change after an
evaluation/improvement cycle. The return value should be a policy in the format
of `pi` from the assignment-wide conventions.

PI always generates a greedy policy, so each action distribution (row of pi)
should be "one-hot", i.e. all zeros except for a single one.
"""
def policy_iteration(P, r, gamma, PE_iters):
    """Performs policy iteration, stopping when the policy does not change."""
    return np.zeros(r.shape)  # TODO: Implement.

