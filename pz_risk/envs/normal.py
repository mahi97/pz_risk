from risk_env import *
from register import register


class Normal(RiskEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, n_agent=6, board_name='world'):
        super().__init__(
            n_agent=n_agent,
            board_name=board_name
        )


class NormalEnv2(Normal):
    def __init__(self):
        super().__init__(n_agent=2)


class NormalEnv4(Normal):
    def __init__(self):
        super().__init__(n_agent=4)


class NormalEnv6(Normal):
    def __init__(self):
        super().__init__(n_agent=6)


register(
    id='Risk-Normal-2-v0',
    entry_point='pz_risk.envs:NormalEnv2'
)

register(
    id='Risk-Normal-4-v0',
    entry_point='pz_risk.envs:NormalEnv4'
)

register(
    id='Risk-Normal-6-v0',
    entry_point='pz_risk.envs:NormalEnv6'
)