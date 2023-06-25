import numpy as np
import yaml
from models.model_abstract import AbstractModel


class FourWheelModel(AbstractModel):
    def __init__(self, dt, open_loop_tf, T_peak, T_slope)-> None:
        
        super().__init__(dt,open_loop_tf, T_peak, T_slope)  # Call the base class constructor
        # Car parameters
        
        # Tire parameters
        
        #stae of the car
        self.state = np.zeros(6) 

        self.slip_angles = np.zeros(4)
        self.slips = []
        

             
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
        SA_fl = np.arctan2(vy + self.car.lf * dyaw, vx - self.car.tw/2 * dyaw) - steering 
        SA_fr = np.arctan2(vy + self.car.lf * dyaw, vx + self.car.tw/2 * dyaw) - steering 
        SA_rl = np.arctan2(vy - self.car.lr * dyaw, vx - self.car.tw/2 * dyaw) 
        SA_rr = np.arctan2(vy - self.car.lr * dyaw, vx + self.car.tw/2 * dyaw)
        current_slip = np.array([SA_fl, SA_fr, SA_rl, SA_rr])
        self.slip_angles = np.row_stack((self.slip_angles, current_slip))
        self.slips.append(SA_fl)
        # Longitudinal tire force
        F_fx = Fx_FL + Fx_FR
        F_rx = Fx_RL + Fx_RR
        # Aero forces
        Fz_lift = 0.5 * self.car.rho * self.car.Cl * self.car.A * vx**2
        Fx_drag = 0.5 * self.car.rho * self.car.Cd * self.car.A * vx**2
        F_res = (self.car.m*self.g + Fz_lift) * self.car.R_res

        #Lateral tyre force
        
        # Accelerations
        ax = (F_rx + F_fx * np.cos(steering) - Fx_drag - F_res) / self.car.m
        ay_ve = (1/20)*vx*vx #(F_rx + F_fx * np.cos(steering) ) / self.car.m
        #ay_ve = ((F_rl_y + F_rr_y) + (F_fl_y + F_fr_y) * np.cos(steering) + F_fx * np.sin(steering)) / self.car.m # NOT SURE ABOUT THIS

        # Online sensor data
        #ay_ve = x_vect[7]

        # Vertical force
        F_fl_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb -
                        self.car.m * ax * self.car.h/self.car.l_wb - Fx_drag * self.car.h/self.car.l_wb)# -self.car.m * ay_ve * self.car.h/self.car.l_wb )
                        
        F_fr_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb -
                        self.car.m * ax * self.car.h/self.car.l_wb - Fx_drag * self.car.h/self.car.l_wb) #+ self.car.m * ay_ve * self.car.h/self.car.l_wb )
                        
        F_rl_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb + 
                        self.car.m * ax * self.car.h/self.car.l_wb + Fx_drag * self.car.h/self.car.l_wb)#- self.car.m * ay_ve * self.car.h/self.car.l_wb )
                        
        F_rr_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb + 
                        self.car.m * ax * self.car.h/self.car.l_wb + Fx_drag * self.car.h/self.car.l_wb)# +self.car.m * ay_ve * self.car.h/self.car.l_wb )
                        
        # Grip coefficients
        mu_fl_y = self.tire.pacejka_numpy(SA_fl, [T_peak_front, T_slope_front],  tire='front')
        mu_fr_y = self.tire.pacejka_numpy(SA_fr, [T_peak_front, T_slope_front],  tire='front')
        mu_rl_y = self.tire.pacejka_numpy(SA_rl, [T_peak_rear, T_slope_rear],  tire='rear')
        mu_rr_y = self.tire.pacejka_numpy(SA_rr, [T_peak_rear, T_slope_rear],  tire='rear')
        # Lateral tire force
        F_fl_y = F_fl_z * mu_fl_y
        F_fr_y = F_fr_z * mu_fr_y
        F_rl_y = F_rl_z * mu_rl_y
        F_rr_y = F_rr_z * mu_rr_y 
        # Moment from torque vectoring
        Mtv = (Fx_FR - Fx_FL) * np.cos(steering) * self.car.tw / 2
        Mtv += (Fx_RR - Fx_RL) * self.car.tw / 2

        return F_fl_y, F_fr_y, F_rl_y, F_rr_y, F_fx, F_rx, Mtv, Fx_drag, F_res



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
        F_fl_y, F_fr_y, F_rl_y, F_rr_y, F_fx, F_rx, Mtv, Fx_drag, F_res = self.compute_forces_numpy(x_vect, inputs)

        # Dynamics equations
        dxdt = np.concatenate(
            (
            [vx * np.cos(yaw) - vy * np.sin(yaw)], #[vx], #
            [vx * np.sin(yaw) + vy * np.cos(yaw)], #[vy], 
            [dyaw],
            [(F_rx + F_fx * np.cos(steering) - (F_fl_y + F_fr_y)  * np.sin(steering) + self.car.m*vy*dyaw - Fx_drag - F_res) / self.car.m],
            [((F_rl_y + F_rr_y) + (F_fl_y + F_fr_y) * np.cos(steering) + F_fx * np.sin(steering) - self.car.m*vx*dyaw) / self.car.m],
            [(-(F_rl_y + F_rr_y) * self.car.lr + (F_fl_y + F_fr_y) * self.car.lf * np.cos(steering) + (F_fl_y - F_fr_y) * self.car.tw / 2 * np.sin(steering) + F_fx * self.car.lf * np.sin(steering) + Mtv) / self.car.Iz]
            )
        )


        return dxdt
    

    
    


