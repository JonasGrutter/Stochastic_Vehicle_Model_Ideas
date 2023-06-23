import numpy as np
import yaml
from models.model_abstract import AbstractModel


class BicycleModel(AbstractModel):
    def __init__(self, dt, open_loop_tf, T_peak, T_front)-> None:
        
        super().__init__(dt,open_loop_tf,T_peak,T_front)  # Call the base class constructor
        #stae of the car
        self.state = np.zeros(6)      
    
    #def compute_forces_numpy(self, x, u, p_sensors):
    def compute_forces_numpy(self, x_vect, inputs):
        # inputs
        steering = inputs[0]
        Fx_FL = inputs[1]
        Fx_FR = inputs[2]
        Fx_RL = inputs[3]
        Fx_RR = inputs[4]
        
        # States at time i
        vx = x_vect[3]
        vy = x_vect[4]
        dyaw = x_vect[5]
        # Control inputs
        T_peak_front = self.tire.T_peak_front
        T_slope_front = self.tire.T_slope_front
        T_peak_rear = self.tire.T_peak_rear
        T_slope_rear = self.tire.T_slope_rear


        

        # Slip angles
        SA_f = np.arctan2(vy + self.car.lf * dyaw, vx) - steering 
        SA_r = np.arctan2(vy - self.car.lr * dyaw, vx)


        # Grip coefficients
        # Grip coefficients
        mu_fy = self.tire.pacejka_numpy(SA_f, [T_peak_front, T_slope_front],  tire='front')
        mu_ry = self.tire.pacejka_numpy(SA_r, [T_peak_rear, T_slope_rear], tire='rear')

        # Longitudinal tire force
        F_fx = Fx_FL + Fx_FR
        F_rx = Fx_RL + Fx_RR
        # Aero forces
        Fz_lift = 0.5 * self.car.rho * self.car.Cl * self.car.A * vx**2
        Fx_drag = 0.5 * self.car.rho * self.car.Cd * self.car.A * vx**2
        F_res = (self.car.m*self.g + Fz_lift) * self.car.R_res
        
        
        # Accelerations
        ax = (F_rx + F_fx * np.cos(steering) - Fx_drag - F_res) / self.car.m
        #therm_for_sol = mu_fy*np.sin(steering)*((self.car.lf/self.car.l_wb)*(self.car.m*self.g+Fz_lift))
        #ax = (F_rx + F_fx * np.cos(steering) - Fx_drag - F_res - therm_for_sol ) / (self.car.m *(1+np.sin(steering)*mu_fy*self.car.h/self.car.l_wb))

        # Vertical force
        F_fz = (self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb - self.car.m * ax * self.car.h/self.car.l_wb - Fx_drag * self.car.h/self.car.l_wb
        F_rz = (self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb + self.car.m * ax * self.car.h/self.car.l_wb + Fx_drag * self.car.h/self.car.l_wb
                        


        # Lateral tire force
        F_fy = F_fz * mu_fy
        F_ry = F_rz * mu_ry 
        # Moment from torque vectoring
        Mtv = (Fx_FR - Fx_FL) * np.cos(steering) * self.car.tw / 2
        Mtv += (Fx_RR - Fx_RL) * self.car.tw / 2

        return F_fy, F_ry, F_fx, F_rx, Mtv, Fx_drag, F_res
    




    def get_dynamics_numpy(self, x_vect, inputs):
        # inputs
        steering = inputs[0]
        # States at time i
        x = x_vect[0]
        y = x_vect[1]
        yaw = x_vect[2]
        
        vx = x_vect[3]
        vy = x_vect[4]
        dyaw = x_vect[5]
        # Steering action
        # Compute tire forces and moment
        F_fy, F_ry, F_fx, F_rx, Mtv, Fx_drag, F_res = self.compute_forces_numpy(x_vect, inputs)

        # Dynamics equations
        dxdt = np.concatenate(
            (
            [vx * np.cos(yaw) - vy * np.sin(yaw)], 
            [vx * np.sin(yaw) + vy * np.cos(yaw)], 
            [dyaw],
            [(F_rx + F_fx * np.cos(steering) - F_fy  * np.sin(steering) + self.car.m*vy*dyaw - Fx_drag - F_res) / self.car.m],
            [(F_ry + F_fy * np.cos(steering) + F_fx * np.sin(steering) - self.car.m*vx*dyaw) / self.car.m],
            [(-F_ry * self.car.lr + F_fy * self.car.lf * np.cos(steering) + F_fx * self.car.lf * np.sin(steering) + Mtv) / self.car.Iz]
            )
        )


        return dxdt
    
    
