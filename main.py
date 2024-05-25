from optimization_problems import AckleyFunction
from pso_variants import PSO, PSOInertiaWeightAdjustment
from topologies import GlobalBestTopology, RingTopology
from metrics_and_visualization import run_optimization, plot_metrics

if __name__ == "__main__":
    problem = AckleyFunction(2)