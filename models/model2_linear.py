import numpy as np
import yaml
from models.model_abstract import AbstractModel

class Dynamic_Bicycle_Linear(AbstractModel):
    def __init__(self, velocity, dt, open_loop_tf, T_peak, T_slope) -> None:
        
        # Call the constructor of AbstractModel
        super().__init__(dt,open_loop_tf, T_peak, T_slope)
        
        self.state = np.zeros(6)
        self.state[3] = velocity
        
        self.alpha_r = 0
        self.alpha_f = 0
        
        self.Fry = 0
        self.Ffy = 0

        self.vx_dot = 0
        self.vy_dot = 0
        self.vtheta_dot = 0
        


    def update_fw_euler(self, steering_angle, dt):
        
        alpha_r, alpha_f = self.get_slip_angles(steering_angle)

        Fry, Ffy = self.get_forces(alpha_r, alpha_f)

        vx_dot = 0
        vy_dot = (1/self.car.m)*(Fry + Ffy *np.cos(steering_angle) - self.car.m*self.state[3]*self.state[5])
        vtheta_dot = (1/self.Iz)*(-self.lr*Fry+self.lf*Ffy*np.cos(steering_angle))

        
        self.alpha_f = alpha_f
        self.alpha_r = alpha_r
        
        self.Ffy = Ffy
        self.Fry = Fry

        self.vx_dot = vx_dot
        self.vy_dot = vy_dot
        self.vtheta_dot = vtheta_dot

        self.state[3] += vx_dot*dt
        self.state[4] += vy_dot*dt
        self.state[5] += vtheta_dot*dt

        xdot = self.state[3]*np.cos(self.state[2])-self.state[4]*np.cos(self.state[2])
        ydot = self.state[3]*np.sin(self.state[2])+self.state[4]*np.cos(self.state[2])
        thetadot = self.state[5]

        self.state[0] += xdot*dt
        self.state[1] += ydot*dt
        self.state[2] += thetadot*dt

    def get_dynamics_numpy(self, x_vect, steering_angle):

        alpha_r, alpha_f = self.get_slip_angles(steering_angle)

        Fry, Ffy = self.get_forces(alpha_r, alpha_f)

        x = x_vect[0]
        y = x_vect[1]
        yaw = x_vect[2]
        
        vx = x_vect[3]
        vy = x_vect[4]
        dyaw = x_vect[5]

        xdot = vx*np.cos(yaw)-vy*np.cos(yaw)
        ydot = vx*np.sin(yaw)+vy*np.cos(yaw)
        
        vx_dot = 0
        vy_dot = (1/self.car.m)*(Fry + Ffy *np.cos(steering_angle) - self.car.m*vx*dyaw)
        vyaw_dot = (1/self.car.Iz)*(-self.car.lr*Fry+self.car.lf*Ffy*np.cos(steering_angle))

        dxdt = np.concatenate(
            (
            [xdot],
            [ydot],
            [dyaw],
            [vx_dot],
            [vy_dot],
            [vyaw_dot]
            )
        )
        return dxdt
    

    def do_open_loop_sim_cst_inputs(self, t0, inputs):
       t,x = super().do_open_loop_sim_cst_inputs(t0, inputs[0]) #Take steering only as Torques are not needed
       return t,x

    
    def get_slip_angles(self, steering_angle):
        
        vx = self.state[3]
        vy = self.state[4]
        dyaw = self.state[5]

        alpha_r = np.arctan2((vy-self.car.lr *dyaw),vx)

        alpha_f = np.arctan2((vy+self.car.lf *dyaw),vx)- steering_angle
        
        return alpha_r, alpha_f

    def get_forces(self, alpha_r, alpha_f):
        Fry =  -self.tire.Cornering_stifness_rear*alpha_r*self.car.m
        Ffy =  -self.tire.Cornering_stifness_front*alpha_f*self.car.m
        
        return Fry, Ffy
