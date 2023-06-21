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
        while t[-1] < self.open_loop_tf-1:
            
            self.t.append(self.t[-1] + self.dt)

            inputs =  [steering_angle[index], Fx_fl[index], Fx_fr[index], Fx_rl[index], Fx_rr[index]]
            # Do 1 step
            self.RK45_1step(self.dt, inputs)
            # Update time and state
            t.append(t[-1] + self.dt)
            x = np.row_stack((x, self.state))
            index += 1


        return t,x
    


    def do_open_loop_sim_from_fsg(self, file_path):
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
            while t[-1] < self.open_loop_tf-1:
                
                self.t.append(self.t[-1] + self.dt)

                inputs =  [steering_angle[index], Fx_fl[index], Fx_fr[index], Fx_rl[index], Fx_rr[index]]
                # Do 1 step
                self.RK45_1step(self.dt, inputs)
                # Update time and state
                t.append(t[-1] + self.dt)
                x = np.row_stack((x, self.state))
                index += 1


            return t,x
    
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
