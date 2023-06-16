import numpy as np
import yaml
from models.model_abstract import AbstractModel


class FourWheelModel_load_transfer(AbstractModel):
    def __init__(self, dt, open_loop_tf, T_peak, T_slope)-> None:
        
        super().__init__(dt,open_loop_tf, T_peak, T_slope)  # Call the base class constructor
        # Car parameters
        
        # Tire parameters
        
        #stae of the car
        self.state = np.zeros(6) 

        self.slip_angles = np.zeros(4)
        

             
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
        SA_fl = np.arctan2(vy + self.car.lf * dyaw, vx ) - steering 
        SA_fr = SA_fl
        SA_rl = np.arctan2(vy - self.car.lr * dyaw, vx ) 
        SA_rr = SA_rl
        
        # Grip coefficients
        mu_fl_y = self.tire.pacejka_numpy(SA_fl, [T_peak_front, T_slope_front],  tire='front')
        mu_fr_y = self.tire.pacejka_numpy(SA_fr, [T_peak_front, T_slope_front],  tire='front')
        mu_rl_y = self.tire.pacejka_numpy(SA_rl, [T_peak_rear, T_slope_rear],  tire='rear')
        mu_rr_y = self.tire.pacejka_numpy(SA_rr, [T_peak_rear, T_slope_rear],  tire='rear')



        current_slip = np.array([SA_fl, SA_fr, SA_rl, SA_rr])
        self.slip_angles = np.row_stack((self.slip_angles, current_slip))
        # Longitudinal tire force
        F_fx = Fx_FL + Fx_FR
        F_rx = Fx_RL + Fx_RR
        # Aero forces
        Fz_lift = 0.5 * self.car.rho * self.car.Cl * self.car.A * vx**2
        Fx_drag = 0.5 * self.car.rho * self.car.Cd * self.car.A * vx**2
        F_res = (self.car.m*self.g + Fz_lift) * self.car.R_res

        #Lateral tyre force
        therm_for_sol = mu_fl_y*np.sin(steering)*((self.car.lf/self.car.l_wb)*(self.car.m*self.g+Fz_lift))
        # Accelerations
        ax = (F_rx + F_fx * np.cos(steering) - Fx_drag - F_res - therm_for_sol ) / (self.car.m *(1+np.sin(steering)*mu_fl_y*self.car.h/self.car.l_wb))
        
        A = mu_fl_y*((self.car.lf/self.car.l_wb)*(self.car.m*self.g+Fz_lift)-self.car.m*ax*(self.car.h/self.car.l_wb)-Fx_drag*(self.car.h/self.car.l_wb))
        B = mu_fr_y*((self.car.lf/self.car.l_wb)*(self.car.m*self.g+Fz_lift)+self.car.m*ax*(self.car.h/self.car.l_wb)-Fx_drag*(self.car.h/self.car.l_wb))
        ay = (1/self.car.m)*(F_fx*np.sin(steering)+A*np.cos(steering)+B)

        '''
        YT_12 = (l_r/l_wb)*(m*g+F_aero)*paj_FL + m*a_X_expl*(l_z/l_wb)*paj_FL;
        YT_34 = (l_f/l_wb)*(m*g+F_aero)*paj_RL - m*a_X_expl*(l_z/l_wb)*paj_RL;
        a_Y_expl = (cos(delta_W)*YT_12 + sin(delta_W)*(F_XT_FL+F_XT_FR) + YT_34)/m;
        
        '''

        '''F_aero = 0.5*roh*coelA*v_X^2
        F_R = 0.5*roh*cdA*v_X^2 + R*(m*g + F_aero)
        A = (F_XT_FL+F_XT_FR)*cos(delta_W) + F_XT_RL+F_XT_RR
        a_X_expl = (A-F_R-sin(delta_W)*(l_r/l_wb)*(m*g+F_aero)*paj_FL)/(m*(1+sin(delta_W)*(l_z/l_wb)*paj_FL))

        A = F_rx + F_fx * np.cos(steering)
        F_R = F_res+Fx_drag
        ax = (A-F_R-np.sin(steering)*(self.car.lf/self.car.l_wb)*(self.car.m*self.g+Fz_lift)*mu_fl_y)/(self.car.m*(1+np.sin(steering)*(self.car.h/self.car.l_wb)*mu_fl_y))'''

        # Vertical force
        F_fl_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb -
                        self.car.m * ax * self.car.h/self.car.l_wb - Fx_drag * self.car.h/self.car.l_wb -self.car.m * ay * self.car.h/self.car.l_wb )
                        
        F_fr_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb -
                        self.car.m * ax * self.car.h/self.car.l_wb - Fx_drag * self.car.h/self.car.l_wb + self.car.m * ay * self.car.h/self.car.l_wb)
                        
        F_rl_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb + self.car.m * ax * self.car.h/self.car.l_wb + Fx_drag * self.car.h/self.car.l_wb - self.car.m * ay * self.car.h/self.car.l_wb )
                        
        F_rr_z = 0.5 * ((self.car.m*self.g + Fz_lift) * self.car.lr/self.car.l_wb + 
                        self.car.m * ax * self.car.h/self.car.l_wb + Fx_drag * self.car.h/self.car.l_wb +self.car.m * ay * self.car.h/self.car.l_wb)
                        
        
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
    

    
    


