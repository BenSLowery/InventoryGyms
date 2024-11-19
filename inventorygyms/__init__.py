from gymnasium.envs.registration import register

register(
    id='inventorygyms/TwoEchelonPLS-v0',
    entry_point='inventorygyms.envs:TwoEchelonPLS'
)
register(
    id='inventorygyms/TwoEchelonPLSTS-v0',
    entry_point='inventorygyms.envs:TwoEchelonPLSTS'
)