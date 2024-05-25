import matplotlib.pyplot as plt
import numpy as np
import distinctipy


def plot_metrics(metrics_data_list):
    plt.figure()
    colors = distinctipy.get_colors(len(metrics_data_list))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    for i, metrics_data in enumerate(metrics_data_list):
        color = colors[i]
        label = f"Run {i + 1}"

        # Plot best fitness value vs iterations
        ax[0, 0].plot(metrics_data["iterations"],
                      metrics_data["best_fitness"], label=label, color=color)

        # Plot distance to global optimum vs iterations
        ax[0, 1].plot(metrics_data["iterations"],
                      metrics_data["distance_to_optimum"], label=label, color=color)

        # Plot standard deviation of particle positions vs iterations
        ax[1, 0].plot(metrics_data["iterations"],
                      metrics_data["std_dev_positions"], label=label, color=color)

        # Plot mean velocity vs iterations
        ax[1, 1].plot(metrics_data["iterations"],
                      metrics_data["mean_velocity"], label=label, color=color)

    ax[0, 0].set_title("Best Fitness vs Iterations")
    ax[0, 0].set_xlabel("Iterations")
    ax[0, 0].set_ylabel("Best Fitness")

    ax[0, 1].set_title("Distance to Global Optimum vs Iterations")
    ax[0, 1].set_xlabel("Iterations")
    ax[0, 1].set_ylabel("Distance to Optimum")

    ax[1, 0].set_title(
        "Standard Deviation of Particle Positions vs Iterations")
    ax[1, 0].set_xlabel("Iterations")
    ax[1, 0].set_ylabel("Std Dev of Positions")

    ax[1, 1].set_title("Mean Velocity vs Iterations")
    ax[1, 1].set_xlabel("Iterations")
    ax[1, 1].set_ylabel("Mean Velocity")

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    plt.tight_layout()
