
class GlobalBestTopology:
    @staticmethod
    def get_best_position(pso, particle_index):
        return pso.global_best_position


class RingTopology:
    @staticmethod
    def get_best_position(pso, particle_index):
        neighbors = list(range(particle_index - pso.num_neighbors,
                         particle_index + pso.num_neighbors + 1))
        neighbors = [i % pso.num_particles for i in neighbors]

        best_fitness = float('inf')
        best_position = None
        for i in neighbors:
            if pso.swarm[i].personal_best_fitness < best_fitness:
                best_fitness = pso.swarm[i].personal_best_fitness
                best_position = pso.swarm[i].personal_best_position.copy()
        return best_position


class StarTopology:
    @staticmethod
    def get_best_position(pso, particle_index):
        if particle_index == 0:
            return pso.global_best_position
        else:
            return pso.swarm[0].personal_best_position


class RandomNeighbourhoodConnectivity:
    @staticmethod
    def get_best_position(pso, particle_index):
        neighbors = pso.random_neighbourhood_connectivity[particle_index]

        best_fitness = float('inf')
        best_position = None
        for i in neighbors:
            if pso.swarm[i].personal_best_fitness < best_fitness:
                best_fitness = pso.swarm[i].personal_best_fitness
                best_position = pso.swarm[i].personal_best_position.copy()
        return best_position
