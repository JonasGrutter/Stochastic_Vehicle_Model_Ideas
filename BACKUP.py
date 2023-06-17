##FINAL VERSION
np.random.seed(2)



class Particle:
    def __init__(self, position, velocity, id):
        self.position = position.copy()
        self.velocity = velocity
        self.best_position = position
        
        self.position_cost_value = self.compute_fitness_function()
        self.best_position_cost_value = self.position_cost_value
        self.id = id
        '''print("Initial position of particle", self.id, self.position )
        print("Initial best position of particle in init", self.id, self.best_position )
        print("Initial best cost of particle", self.id, self.best_position_cost_value )'''

    def update_velocity(self, inertia, cognitive_rate, social_rate, global_best_position):
        r1, r2 = np.random.rand(2)
        cognitive_component = cognitive_rate * r1 * (self.best_position - self.position)
        social_component = social_rate * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_component + social_component

    def update_position(self, lower_bounds, upper_bounds):
        #print("Initial best position of particle in update_pos befofre updating pos", self.id, self.best_position )
        self.position += self.velocity
        #print("Initial best position of particle in update_pos after updating pos", self.id, self.best_position )
        self.position = np.clip(self.position, lower_bounds, upper_bounds)
        self.position_cost_value = self.compute_fitness_function()
        #print("Initial best position of particle in update_pos end of function", self.id, self.best_position )

    def update_best_position(self, objective_function):
        '''print('FOR PARTICLE', self.id)
        print('IN CLASS PARTICLE BEFORE UPDATE', objective_function(self.best_position) == self.best_position_cost_value)
        print('IN CLASS PARTICLE BEFORE POS',self.position )
        print('IN CLASS PARTICLE BEFORE COST',self.position_cost_value )
        print('IN CLASS PARTICLE BEFORE BEST POS',self.best_position )
        print('IN CLASS PARTICLE BEFORE BEST POS VALUE',self.best_position_cost_value )
        print('IN CLASS PARTICLE BEFORE BEST POS EVALUATED', objective_function(self.best_position) )'''
        if self.position_cost_value < self.best_position_cost_value: #objective_function(self.best_position): #self.best_position_cost_value:
            #print('IN IF CONDITION')
            self.best_position = self.position
            self.best_position_cost_value = self.position_cost_value#self.position_cost_value
        '''print('IN CLASS PARTICLE AFTER UPDATE', objective_function(self.best_position) == self.best_position_cost_value)
        print('IN CLASS PARTICLE AFTER BEST POS',self.best_position )
        print('IN CLASS PARTICLE AFTER BEST POS VALUE',self.best_position_cost_value )
        print('IN CLASS PARTICLE AFTER BEST POS EVALUATED', objective_function(self.best_position) )'''



        
    def compute_fitness_function(self):
        return float(self.position[0]**2 + self.position[1]**2)




class PSO:
    def __init__(self, num_particles, max_iterations, lower_bounds, upper_bounds):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.particles = []
        self.global_best_position = None
        self.particle_positions_history = []
        self.particle_fitness_history = []

    def optimize(self, objective_function, inertia=0.5, cognitive_rate=0.5, social_rate=0.5):
        self.initialize_particles()

        for iter in range(self.max_iterations):
            print('Iter', iter)
            
            self.particle_positions_history.append([ particle.position for particle in self.particles])
            self.particle_fitness_history.append([particle.position_cost_value for particle in self.particles])

            for particle in self.particles:
                particle.update_velocity(inertia, cognitive_rate, social_rate, self.global_best_position)
                particle.update_position(self.lower_bounds, self.upper_bounds)
                particle.update_best_position(objective_function)

            self.update_global_best_position(objective_function)

    def initialize_particles(self):
        self.particles = []
        for i in range(self.num_particles):
            position = np.random.uniform(self.lower_bounds, self.upper_bounds)
            velocity = np.zeros_like(position)
            id = i
            particle = Particle(position, velocity, id)
            self.particles.append(particle)

        index_initial_particle_lowest_cost = np.argmin([particle.position_cost_value for particle in self.particles])
        #self.global_best_position = self.particles[index_initial_particle_lowest_cost].position
        self.global_best_position = self.particles[index_initial_particle_lowest_cost].position.copy()
        self.global_best_position_value = self.particles[index_initial_particle_lowest_cost].position_cost_value
        
        print('BEst initial position:', self.global_best_position)
        print('BEst initial cost:', objective_function( self.global_best_position))

    def update_global_best_position(self, objective_function):
        counter = 0
        for particle in self.particles:
            print(self.global_best_position_value == objective_function(self.global_best_position))
            if particle.position_cost_value < self.global_best_position_value: #(self.global_best_position): #objective_function(self.global_best_position):
                #print(particle.position)
                #print(self.global_best_position)
                #print('Global best value before update', self.global_best_position_value)
                #print('Global best position before update', self.global_best_position)
                self.global_best_position = particle.position
                self.global_best_position_value = particle.position_cost_value
                #print('Global best value after update',  self.global_best_position_value)
                #print('Global best position after update', self.global_best_position)
                counter +=1
        print(counter)

    def plot_particle_positions(self, frame):
        plt.cla()
        plt.plot(*zip(*self.particle_positions_history[frame]), 'go', markersize=6)
        plt.contourf(self.X, self.Y, self.Z, levels=np.linspace(0, 100, 50), cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('y')

    def plot_particle_fitness_history(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        # Plot histogram at the start
        ax1.hist(self.particle_fitness_history[0], bins=20, color='blue', edgecolor='black')
        ax1.set_xlabel('Fitness')
        ax1.set_ylabel('Count')
        ax1.set_title('Start of Optimization')

        # Plot histogram at the middle
        middle_index = len(self.particle_fitness_history) // 2
        ax2.hist(self.particle_fitness_history[middle_index], bins=20, color='green', edgecolor='black')
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Count')
        ax2.set_title('Middle of Optimization')

        # Plot histogram at the end
        ax3.hist(self.particle_fitness_history[-1], bins=20, color='red', edgecolor='black')
        ax3.set_xlabel('Fitness')
        ax3.set_ylabel('Count')
        ax3.set_title('End of Optimization')

        plt.tight_layout()
        plt.show()
    def plot_particle_history(self):
        self.particle_positions_history = np.array(self.particle_positions_history)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        # Plot histogram at the start
        ax1.hist2d(self.particle_positions_history[0][:, 0], self.particle_positions_history[0][:, 1], bins=20, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Start of Optimization')

        # Plot histogram at the middle
        middle_index = len(self.particle_positions_history) // 2
        ax2.hist2d(self.particle_positions_history[middle_index][:, 0],self.particle_positions_history[middle_index][:, 1], bins=20, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Middle of Optimization')

        # Plot histogram at the end
        ax3.hist2d(self.particle_positions_history[-1][:, 0], self.particle_positions_history[-1][:, 1], bins=20, cmap='viridis')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('End of Optimization')

        plt.tight_layout()
        plt.show()

    def plot_particle_positions_animate(self):
        self.X, self.Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        self.Z = self.X**2 + self.Y**2

        fig = plt.figure()
        ani = FuncAnimation(fig, self.plot_particle_positions, frames=len(self.particle_positions_history), interval=200)
        plt.show()


# Example usage
pso = PSO(num_particles=100, max_iterations=1000, lower_bounds=[-10, -10], upper_bounds=[10, 10])

def objective_function(x):
    return float(x[0]**2 + x[1]**2)

pso.optimize(objective_function)
pso.plot_particle_history()
pso.plot_particle_fitness_history()
#pso.plot_particle_positions_history()
