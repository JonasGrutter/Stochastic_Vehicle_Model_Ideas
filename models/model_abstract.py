import numpy as np
import yaml
import matplotlib.pyplot as plt

class AbstractModel:
    def __init__(self, dt, open_loop_tf, T_peak, T_slope) -> None:
        self.dt = dt
        self.open_loop_tf = open_loop_tf
        self.t = [0]
        self.tire = Tire(T_peak, T_slope)
        self.car = Car()
        self.g = 9.81

        

    # Get dynamics numpy is needed for every one

    def RK45_1step(self, dt, inputs):
            
            # RK intermadiate vectors
            k1 = self.get_dynamics_numpy(self.state, inputs)
            k2 = self.get_dynamics_numpy(self.state + 0.5 * dt * k1, inputs)
            k3 = self.get_dynamics_numpy(self.state + 0.5 * dt * k2, inputs)
            k4 = self.get_dynamics_numpy(self.state + dt * k3, inputs)
            # State update
            self.state = self.state + (1.0 / 6.0) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    
    def do_open_loop_sim_cst_inputs(self, t0, inputs):
        # recap of times and states
        t = [t0]
        self.state = np.array([0,0,0, 4, 0.5,1.0])
        x = self.state
        
        # While loop
        while t[-1] < self.open_loop_tf:
            
            self.t.append(self.t[-1] + self.dt)

            # Do 1 step
            self.RK45_1step(self.dt, inputs)
            # Update time and state
            t.append(t[-1] + self.dt)
            x = np.row_stack((x, self.state))


        return t,x
    
    # Inputs are from a csv file
    def do_open_loop_sim_from_csv(self, file_path):
        # recap of times and states
        t = [0]
        x = self.state
        # Load the CSV file
        data = np.genfromtxt(file_path, delimiter=',', skip_header=0, names=True, dtype=None)
        
        # Access each column by its tag
        Fx_fl = data['Fx_fl']
        Fx_fr = data['Fx_fr']
        Fx_rl = data['Fx_rl']
        Fx_rr = data['Fx_rr']
        steering_angle = data['steering_angle']

   
        
        # While loop
        index = 0
        while t[-1] < self.open_loop_tf-1:#-1
            
            self.t.append(self.t[-1] + self.dt)

            inputs =  [steering_angle[index], Fx_fl[index], Fx_fr[index], Fx_rl[index], Fx_rr[index]]
            # Do 1 step
            self.RK45_1step(self.dt, inputs)
            # Update time and state
            t.append(t[-1] + self.dt)
            x = np.row_stack((x, self.state))
            index += 1


        return t,x
    

    def do_open_loop_sim_from_np_array(self, command_array, state_array):
        # recap of times and states
        t = [0]
        self.state = np.array([state_array[0,0], state_array[0,1], state_array[0,2], state_array[0,3], state_array[0,4], state_array[0,5]])
        x = self.state
        
        # Access each column by its tag
        Fx_fl = command_array[:,0]
        Fx_fr = command_array[:,1]
        Fx_rl = command_array[:,2]
        Fx_rr = command_array[:,3]
        steering_angle = command_array[:,4]

   
        
        # While loop
        index = 0
        while t[-1] < self.open_loop_tf-2*self.dt:#-1
                   
            self.t.append(self.t[-1] + self.dt)

            inputs =  [steering_angle[index], Fx_fl[index], Fx_fr[index], Fx_rl[index], Fx_rr[index]]
            # Do 1 step
            self.RK45_1step(self.dt, inputs)
            # Update time and state
            t.append(t[-1] + self.dt)
            x = np.row_stack((x, self.state))
            index += 1


        return t,x
    
    def store_amzsim_manoeuvers_position(self):
        file_path = '/Users/jonas/Desktop/intergation_model/open_loop_inputs/car_command_for_sim_PLEASE.csv'
        t,data = self.do_open_loop_sim_from_csv(file_path)

        header = 'x,y,yaw,vx,vy,dyaw'  # Column names
        np.savetxt('/Users/jonas/Desktop/intergation_model/open_loop_inputs/ref_trajectory.csv', data, delimiter=',', header=header, comments='')
        print('Reference trajectories for amzsim manoeuvers has been saved!')


    def do_open_loop_sim_from_amzsim_maoeuvers(self, chunk_size):
            # recap of times and states
            t = [0]
            
            # Load the CSV file
            file_path_commands = '/Users/jonas/Desktop/intergation_model/open_loop_inputs/car_command_for_sim_PLEASE.csv'
            commands = np.genfromtxt(file_path_commands, delimiter=',', skip_header=0, names=True, dtype=None)
            
            file_path_ref_traj = '/Users/jonas/Desktop/intergation_model/open_loop_inputs/ref_trajectory.csv'
            ref_traj = np.genfromtxt(file_path_ref_traj, delimiter=',', skip_header=0, names=True, dtype=None)
            
            # Access each column by its tag
            Fx_fl = commands['Fx_fl']
            Fx_fr = commands['Fx_fr']
            Fx_rl = commands['Fx_rl']
            Fx_rr = commands['Fx_rr']
            steering_angle = commands['steering_angle']

            x_ref = ref_traj['x']
            y_ref = ref_traj['y']
            yaw_ref = ref_traj['yaw']
            
            vx_ref = ref_traj['vx']
            vy_ref = ref_traj['vy']
            dyaw_ref = ref_traj['dyaw']

            ref_traj = np.hstack([x_ref.reshape(-1,1), y_ref.reshape(-1,1), yaw_ref.reshape(-1,1), vx_ref.reshape(-1,1), vy_ref.reshape(-1,1), dyaw_ref.reshape(-1,1)])
            
            
            # While loop
            index = 0
            self.state = np.array([x_ref[0],y_ref[0],yaw_ref[0],vx_ref[0],vy_ref[0],dyaw_ref[0]])
            x = self.state
            while t[-1] < self.open_loop_tf-2*self.dt:
                
                if index%chunk_size == 0 and index !=0:
                    self.state = np.array([x_ref[index],y_ref[index],yaw_ref[index],vx_ref[index],vy_ref[index],dyaw_ref[index]])

                self.t.append(self.t[-1] + self.dt)

                inputs =  [steering_angle[index], Fx_fl[index], Fx_fr[index], Fx_rl[index], Fx_rr[index]]
                # Do 1 step
                self.RK45_1step(self.dt, inputs)
                # Update time and state
                t.append(t[-1] + self.dt)
                x = np.row_stack((x, self.state))
                index += 1
            
            l2_norm_position = np.linalg.norm(ref_traj[:,:2] - x[:,:2], axis=1).sum()
            l2_norm_velocity = np.linalg.norm(ref_traj[:,2:] - x[:,2:], axis=1).sum()

            w1 = 0.5
            w2 = 0.5
            kpi = w1*l2_norm_position+w2*l2_norm_velocity

            return t,x,kpi
    
    def kinematik_model_radius(self, steering_angle):
        L = self.car.lf + self.car.lr
        radius = L/(np.tan(steering_angle)*np.cos(np.arctan((self.car.lr/L)*np.tan(steering_angle))))
        return  radius



class Tire:
    def __init__(self, T_peak, T_slope) -> None:
        # Open the YAML file
        with open('./config/config.yaml', 'r') as file:
            # Load the YAML content
            data = yaml.safe_load(file)
        tire_params = data['tire_params']
        self.Bf = tire_params['Bf']
        self.Cf = tire_params['Cf']
        self.Df = tire_params['Df']
        self.Ef = tire_params['Ef']
        self.Hf = tire_params['Hf']
        self.Vf = tire_params['Vf']
        self.Br = tire_params['Br']
        self.Cr = tire_params['Cr']
        self.Dr = tire_params['Dr']
        self.Er = tire_params['Er']
        self.Hr = tire_params['Hr']
        self.Vr = tire_params['Vr']

        # Scaling factors
        self.T_peak_front = T_peak #tire_params['T_peak_front']
        self.T_slope_front = T_slope #tire_params['T_slope_front']
        self.T_peak_rear = T_peak #tire_params['T_peak_rear']
        self.T_slope_rear = T_slope #tire_params['T_slope_rear']

        self.Cornering_stifness_rear = -self.Cf*self.Bf*self.Df*self.T_peak_front*self.T_slope_front
        self.Cornering_stifness_front = -self.Cr*self.Br*self.Dr*self.T_peak_rear*self.T_slope_rear
        
    def pacejka_numpy(self, SA, u, tire):
        # Tire scaling params
        T_peak = u[0]
        T_slope = u[1]
        # Prior tire params
        if tire == 'front':
            B = self.Bf * self.T_slope_front
            C = self.Cf
            D = self.Df * self.T_peak_front
            E = self.Ef
            H = self.Hf
            V = self.Vf
        elif tire == 'rear':
            B = self.Br * self.T_slope_rear
            C = self.Cr
            D = self.Dr * self.T_peak_rear
            E = self.Er
            H = self.Hr
            V = self.Vr
        else:
            raise ValueError('tire was not correctly specified: should be either front or rear')


        mu_y = D * np.sin(C * np.arctan(B * (SA + H))) + V
        
        return mu_y

    def pacejka_numpy_complicated(self, SA, u, tire):
        # Tire scaling params
        T_peak = u[0]
        T_slope = u[1]
        # Prior tire params
        if tire == 'front':
            B = self.Bf * self.T_slope_front
            C = self.Cf
            D = self.Df * self.T_peak_front
            E = self.Ef
            H = self.Hf
            V = self.Vf
        elif tire == 'rear':
            B = self.Br * self.T_slope_rear
            C = self.Cr
            D = self.Dr * self.T_peak_rear
            E = self.Er
            H = self.Hr
            V = self.Vr
        else:
            raise ValueError('tire was not correctly specified: should be either front or rear')


        mu_y = D * np.sin(C * np.arctan(B * (SA + H) - E * (B * (SA + H) + np.arctan(B * (SA + H))))) + V
        
        return mu_y
    
    def plot_pacejka(self):
        import numpy as np
        SAS = np.linspace(-0.2, 0.2, 200)
        mus = np.empty(SAS.size)
        for i in range(SAS.size):
            mus[i] = self.pacejka_numpy(SAS[i], [0,0], 'front')
        plt.plot(SAS, mus, label='MF')
        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\mu_y$')
        plt.grid(True)
        plt.show()


    def plot_pacejka_complicated(self):
        import numpy as np
        SAS_2 = np.linspace(-0.2, 0.2, 200)
        mus_2 = np.empty(SAS_2.size)
        for i in range(SAS_2.size):
            mus_2[i] = self.pacejka_numpy_complicated(SAS_2[i], [0,0], 'front')
        plt.plot(SAS_2, mus_2, label='MF')
        plt.legend()
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\mu_y$')
        plt.grid(True)
        plt.show()
        


class Car:
    def __init__(self) -> None:
            # Open the YAML file
        with open('./config/config.yaml', 'r') as file:
            # Load the YAML content
            data = yaml.safe_load(file)
        car_params = data['car_params']
        # Physical attributes
        self.m = car_params['m'] # Total mass
        self.Iz = car_params['Iz'] # Yaw inertia
        self.lf = car_params['lf'] # Front wheelbase
        self.lr = car_params['lr'] # Rear wheelbase
        self.l_wb = car_params['l_wb'] # Total wheelbase
        self.tw = car_params['tw'] # Trackwidth
        self.r_tire = car_params['r_tire'] # Loaded tire radius
        self.rho = car_params['rho'] # Air density
        self.Cd = car_params['Cd'] # Drag coefficient
        self.Cl = car_params['Cl'] # Lift coefficient
        self.A =  car_params['A'] # Aero area
        self.h = car_params['h'] # COP height
        self.R_res = car_params['R_res'] # Rolling resistance
        self.gr_w = car_params['gr_w'] # Wheel gear ratio
        self.Iw = car_params['Iw'] # Wheel inertia

    '''def kinematik_model_radius(self, steering_angle):
        L = self.lf + self.lr
        radius = L/(np.tan(steering_angle)*np.cos(np.arctan((self.lr/L)*np.tan(steering_angle))))
        return  radius'''
