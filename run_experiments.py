import random
import numpy as np
import matplotlib.pyplot as plt
import metrics_and_visualization as mv
from optimization_problems import AckleyFunction
from pso_variants import PSO, PSOWithWeight
import os


def run_experiment(swarm_size, pso_type="standard", topology_type="gbest", file_name="metrics_std_pso.png"):
    num_particles = swarm_size
    num_dimensions = 10
    position_range = 30
    velocity_range = 30

    all_metrics_data = []

    for run in range(10):
        if pso_type == "standard":
            pso = PSO(AckleyFunction(dimensions=10), num_particles=swarm_size,
                      topology=topology_type, max_iterations=1000, pso_type=pso_type)
        else:
            pso = PSOWithWeight(AckleyFunction(dimensions=10), num_particles=swarm_size,
                                topology=topology_type, max_iterations=1000, pso_type=pso_type)

        i = pso.optimize()

        metrics_data = {
            "iterations": i,
            "best_fitness": pso.global_fitness,
            "distance_to_optimum": pso.swarm_centre_of_mass,
            "std_dev_positions": pso.standard_deviation,
            "mean_velocity": pso.velocity_vector_length
        }

        all_metrics_data.append(metrics_data)

    mv.plot_metrics(all_metrics_data)

    plt.savefig("results/" + file_name)
    plt.close()


# part6
run_experiment(30, file_name="metrics_std_pso.png")
# run_experiment(30, pso_type="inertia",
#                file_name="metrics_std_pso_weight_adjust.png")


# # part7
run_experiment(20, file_name="metrics_std_pso_swarm20.png")
run_experiment(100, file_name="metrics_std_pso_swarm100.png")
run_experiment(200, file_name="metrics_std_pso_swarm200.png")

# # # part8
run_experiment(30, pso_type="inertia",
               file_name="metrics_std_pso_weight_adjust.png")

# # part9
run_experiment(30, file_name="metrics_std_pso_topo_gbest.png")
run_experiment(30, topology_type="ring",
               file_name="metrics_std_pso_topo_ring.png")
run_experiment(30, topology_type="star",
               file_name="metrics_std_pso_topo_star.png")
run_experiment(30, topology_type="rand",
               file_name="metrics_std_pso_topo_rand.png")

# # part10
# # try 20 gbest inertia
run_experiment(20, topology_type="gbest", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_gbest_swarm20.png")
# # try 100 gbest inertia
run_experiment(100, topology_type="gbest", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_gbest_swarm100.png")
# # try 200 gbest inertia
run_experiment(200, topology_type="gbest", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_gbest_swarm200.png")
# # try 200 gbest standard
run_experiment(200, topology_type="gbest",
               file_name="metrics_std_pso_topo_gbest_swarm200.png")

# # try 20 ring inertia
run_experiment(20, topology_type="ring", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_ring_swarm20.png")
# # try 100 ring inertia
run_experiment(100, topology_type="ring", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_ring_swarm100.png")
# # try 200 ring inertia
run_experiment(200, topology_type="ring", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_ring_swarm200.png")
# # try 200 ring standard
run_experiment(200, topology_type="ring",
               file_name="metrics_std_pso_topo_ring_swarm200.png")

# # try 20 star inertia
run_experiment(20, topology_type="star", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_star_swarm20.png")
# # try 100 star inertia
run_experiment(100, topology_type="star", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_star_swarm20.png")
# # try 200 star inertia
run_experiment(200, topology_type="star", pso_type="inertia",
               file_name="metrics_weight_adjust_pso_topo_star_swarm200.png")
# # try 200 star standard
run_experiment(200, topology_type="star",
               file_name="metrics_std_pso_topo_star_swarm200.png")

# try 20 rand inertia
run_experiment(20, topology_type="rand", pso_type="inertia",
               file_name="metrics_weight_adjiust_pso_topo_rand_swarm20.png")
# # try 100 rand inertia
run_experiment(100, topology_type="rand", pso_type="inertia",
               file_name="metrics_weight_adjiust_pso_topo_rand_swarm100.png")
# # try 200 rand inertia
run_experiment(200, topology_type="rand", pso_type="inertia",
               file_name="metrics_weight_adjiust_pso_topo_rand_swarm200.png")
# # try 200 rand standard
run_experiment(200, topology_type="rand",
               file_name="metrics_std_pso_topo_rand_swarm200.png")
