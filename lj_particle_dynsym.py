import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime 
import copy

class LJParticleDynamics2D:
    def __init__(self, particle_num=20, temperature=0.01, box_size_x=40, box_size_y=40, time_step=0.001):
        self.particle_num = particle_num
        self.rand_num = 0.1
        self.temperature = temperature
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.time_step = time_step
        
        self.positions = None
        self.momenta = None
        
        self.prev_positions = None
        self.prev_momenta = None
        self.prev_force = None
        
        self.sigma = None
        self.epsilon = None
        self.mass = None
        self.steps = None
        
        self.LJ_energy = None
        self.kinetic_energy = None
        self.total_energy = None
        self.prev_LJ_force = None
        self.LJ_force = np.zeros((self.particle_num, 2))
        
        self.init_positions()
        self.init_momenta()
        self.LJ_order = 6.0
        
    def calc_LJ_force(self):
        LJ_force = np.zeros((self.particle_num, 2))
        for i in range(self.particle_num):
            
            for j in range(self.particle_num):
                if i < j:
                    x = self.positions[i][0] - self.positions[j][0] + 1e-10
                    y = self.positions[i][1] - self.positions[j][1] + 1e-10
                    
                    r = np.array([x, y])
                    sigma = (self.sigma[i] + self.sigma[j]) / 2
                    eps = np.sqrt(self.epsilon[i] * self.epsilon[j])
                    f = -2.0 * eps * self.LJ_order * (sigma ** (2.0 * self.LJ_order) * np.linalg.norm(r) ** (-2.0 * self.LJ_order - 1.0) - sigma ** (self.LJ_order) * np.linalg.norm(r) ** (-1.0 * self.LJ_order - 1.0))
                    LJ_force[i][0] += -1.0 * f * x / np.linalg.norm(r) 
                    LJ_force[i][1] += -1.0 * f * y / np.linalg.norm(r)
                    LJ_force[j][0] -= -1.0 * f * x / np.linalg.norm(r)
                    LJ_force[j][1] -= -1.0 * f * y / np.linalg.norm(r)
                    
        
        self.prev_LJ_force = self.LJ_force 
        self.LJ_force = LJ_force
        return
    
    def calc_total_energy(self):
        self.total_energy = self.kinetic_energy + self.LJ_energy
        return 
    
    
    def calc_kinetic_energy(self):
        
        tmp_kinetic_energy = 0.0
        
        for i in range(self.particle_num):
            tmp_kinetic_energy += np.sum(self.momenta[i] ** 2.0) / (self.mass[i] * 2.0)
        self.kinetic_energy = tmp_kinetic_energy

        return
    
    def calc_temperature(self):
        self.temperature = self.kinetic_energy / self.particle_num 
        return
    
    def calc_LJ_energy(self):
        LJ_energy = 0.0
        for i in range(self.particle_num):
            for j in range(self.particle_num):
                if i < j:
                    r = self.positions[i] - self.positions[j] + 1e-10
                    sigma = (self.sigma[i] + self.sigma[j]) / 2
                    eps = np.sqrt(self.epsilon[i] * self.epsilon[j])
                    LJ_energy += eps * ((sigma/np.linalg.norm(r)) ** (2*self.LJ_order) -2.0 * (sigma/np.linalg.norm(r)) ** self.LJ_order)
                    
        self.LJ_energy = LJ_energy
        return
        

    
    def init_sigma(self):
        self.sigma = np.ones((self.particle_num)) * self.rand_num * 2.0
        return
    
    def init_epsilon(self):
        self.epsilon = np.ones((self.particle_num)) * self.rand_num * 10.0
        return
    
    def init_mass(self):
        self.mass = np.ones((self.particle_num)) * self.rand_num 
        return
    

    def init_positions(self):
        self.positions = np.random.rand(self.particle_num, 2) * np.array([self.box_size_x, self.box_size_y])
        self.prev_positions = self.positions
        return
    
    def init_momenta(self):
        self.momenta =  np.ones((self.particle_num, 2)) * 0.1
        self.prev_momenta = self.momenta 
        return
    
    def init_simulation(self):
        self.init_positions()
        self.init_momenta()
        self.init_sigma()
        self.init_epsilon()
        self.init_mass()
        return
    

    
    
    def main_loop(self, steps=1000):
        self.steps = steps
        print("start")
        potential_list = []
        kinetic_ene_list = []
        tot_ene_list = []
        temperature_list = []
        positions_list = []
        
        self.init_simulation()
        self.calc_LJ_force()
        
        for i in range(steps):
            
            self.momenta += self.time_step * (self.LJ_force + self.prev_LJ_force) / 2.0
            
            tmp_delta_positions = np.zeros((self.particle_num, 2))
            for j in range(self.particle_num):
                tmp_delta_positions[j] += self.time_step / self.mass[j] * (0.5 * self.LJ_force[j] * self.time_step + self.momenta[j])
            
            self.positions += tmp_delta_positions
            self.positions = np.mod(self.positions, np.array([self.box_size_x, self.box_size_y]))
            
            
            self.calc_LJ_energy() #potential energy
            self.calc_LJ_force()
            self.calc_kinetic_energy()
            self.calc_temperature()
            self.calc_total_energy()
            
            
            
            if i % 50 == 0:
                print("# STEP "+str(i))
                potential_list.append(self.LJ_energy)
                kinetic_ene_list.append(self.kinetic_energy)
                tot_ene_list.append(self.total_energy)
                temperature_list.append(self.temperature)
                positions_list.append(self.positions)
                print("LJ Energy: "+str(self.LJ_energy))
                print("Kinetic Energy: "+str(self.kinetic_energy))
                print("Total Energy: "+str(self.total_energy))
                print("Temperature: "+str(self.temperature))
            
        
        potential_list = np.array(potential_list)
        kinetic_ene_list = np.array(kinetic_ene_list)
        tot_ene_list = np.array(tot_ene_list)
        temperature_list = np.array(temperature_list)
        positions_list = np.array(positions_list)
        
        
        VIS = Visualization()
        VIS.visualize_energy(potential_list, kinetic_ene_list, tot_ene_list, temperature_list)
        VIS.animate_positions(positions_list, self.box_size_x, self.box_size_y)
        
        return potential_list, kinetic_ene_list, tot_ene_list, temperature_list, positions_list
        
    
class Visualization:
    def __init__(self):
        dt6= datetime.datetime.now()
        self.strdt6 = dt6.strftime('%Y_%m_%d_%H_%M_%S')
        
        return

    
    def animate_positions(self, positions_list, x, y):
        fig = plt.figure()

        ax = fig.add_subplot(111, aspect=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, x)
        ax.set_ylim(0, y)
        
        def plot(data):
            ax.cla()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(0, x)
            ax.set_ylim(0, y)
            ax.plot(data[:, 0], data[:, 1], c='red', marker='o', linestyle='None')
            
            
        
        ani = animation.FuncAnimation(fig, plot, interval=100, frames=positions_list)
        ani.save(self.strdt6+"_results_ani.gif", writer="imagemagick")
        
        plt.close()
        return
    
    def visualize_energy(self, potential_list, kinetic_ene_list, tot_ene_list, temperature_list):
        fig = plt.figure()
        
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)
        
        x = np.arange(len(potential_list))
        y1 = potential_list
        y2 = kinetic_ene_list
        y3 = tot_ene_list
        y4 = temperature_list
        
        c1 = "blue"
        c2 = "green"
        c3 = "red"
        c4 = "black"
        
        ax1.set_xlabel('time step')  
        ax1.set_ylabel('energy')  
        ax2.set_ylabel('energy')  
        ax3.set_ylabel('energy')  
        ax4.set_ylabel('temperature')  
        
        ax1.plot(x, y1, c=c1, label='Potential Energy')
        ax2.plot(x, y2, c=c2, label='Kinetic Energy')
        ax3.plot(x, y3, c=c3, label='Total Energy')
        #ax3.set_yscale('log')
        ax1.legend(loc=0) 
        ax2.legend(loc=0) 
        ax3.legend(loc=0) 
        ax4.plot(x, y4, c=c4, label='Temperature')
        ax4.legend(loc=0) 
        
        fig.tight_layout()
        
        fig.savefig(self.strdt6+"_results_energy.png")
        plt.close()
                
        return
            
if __name__ == "__main__":
    LJ = LJParticleDynamics2D()
    LJ.main_loop(3000)