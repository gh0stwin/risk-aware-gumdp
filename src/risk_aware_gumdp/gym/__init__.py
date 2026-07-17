from risk_aware_gumdp.gym.grid_exploration_occupancy_env import GridExplorationOccupancyEnv
from risk_aware_gumdp.gym.path_imitation_occupancy_env import PathImitationOccupancyEnv
from risk_aware_gumdp.gym.resource_gathering_occupancy_env import ResourceGatheringOccupancyEnv


def make_gym(id: str, **kwargs):
    return {
        "MaximumStateEntropyExploration-v0": GridExplorationOccupancyEnv,
        "ImitationLearning-v0": PathImitationOccupancyEnv,
        "MultiObjective-v0": ResourceGatheringOccupancyEnv,
    }[id](**kwargs)
