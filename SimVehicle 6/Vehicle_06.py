# 2019 06 19
import numpy as np
from scipy.integrate import odeint
 

class vehicle:
    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.position = np.array([x, y])    # m    
        self.velocity = 0.0                 # m/s
        self.angle = angle                  # rad
        self.heading = self.angle           # rad
        self.L = 0.72                        # m    Wheel base length
        self.Lf = 0.35  #  0.35             # m    CG to front axis length
        self.Lr = self.L-self.Lf            # m    CG to rear axis length

        self.max_velocity = 20              # m/s
        self.max_acceleration = 3.0         # m/s^2
        self.max_jerk = 3.0                 # m/s^3
        self.max_steering = 0.3             # 0.35rad = 20deg  0.3rad = 17deg  0.25rad = 14deg 0.2rad = 11.5deg
        self.max_steer_rate = 1             # rad/s
        #? self.brake_deceleration = 10
        #? self.free_deceleration = 2

        self.position_rear = np.array([-self.Lr*np.cos(self.angle)+self.position[0], -self.Lr*np.sin(self.angle)+self.position[1]])
        self.position_front = np.array([self.Lf*np.cos(self.angle)+self.position[0], self.Lf*np.sin(self.angle)+self.position[1]])
        
        self.acceleration = 0.0
        self.steering = 0.0

        self.mass = 125+24+24   # kg Vehicle + 2*batteries
        self.inertia = 46.33    # kgm^2
        self.C = 1400           # Tyre Cornering stiffness coefficient From tyre data in radians
        self.cg = np.array([0,0.0291])  # m from geometric center
        self.beta = 0           # Slip angle
        self.betadot = 0        # Slip rate
        self.delta = 0          # Steering angle
        self.psi = 0            # Yaw angle
        self.psidot = 0         # Yaw rate
        self.psiddot = 0        # Yaw acceleration
        #? self.tyre_slip = 0



#* Set values from the simulator class
#* Set acceleration cap based on jerk
    def set_acceleration(self, a, dt):
        max_accel_change = self.max_jerk*dt
        if np.abs(a-self.acceleration)>max_accel_change:
            self.acceleration = self.acceleration+np.sign(a-self.acceleration)*max_accel_change
        else:
            self.acceleration = a
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))

#* Set steering cap based on steering rate
    def set_steering(self, steer, dt):
        max_steer_change = self.max_steer_rate*dt
        if np.abs(steer-self.steering)>max_steer_change:
            self.steering = self.steering+np.sign(steer-self.steering)*max_steer_change
        else:
            self.steering = steer
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))



#* Kinematic model - based on rear driving axis
    def update_kr(self, dt):
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))
        
        self.velocity += self.acceleration * dt
        self.velocity = max(0, min(self.velocity, self.max_velocity))

        if self.steering:
            turning_radius = self.L / np.tan(self.steering)
            angular_velocity = self.velocity / turning_radius
        else:
            angular_velocity = 0

        self.angle += angular_velocity * dt
        self.position_rear[0] += self.velocity*np.cos(self.angle) * dt 
        self.position_rear[1] += self.velocity*np.sin(self.angle) * dt
        
        self.heading = self.angle 
        self.position = [self.Lr*np.cos(self.angle)+self.position_rear[0], self.Lr*np.sin(self.angle)+self.position_rear[1]]
        self.position_front = [self.L*np.cos(self.angle)+self.position_rear[0], self.L*np.sin(self.angle)+self.position_rear[1]]

#* Kinematic model - based on vehicle centre
    def update_kc(self, dt):
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))
        self.velocity += self.acceleration * dt
        self.velocity = max(0, min(self.velocity, self.max_velocity))
        
        if self.steering:
            slip_angle = np.arctan((self.Lr*np.tan(self.steering))/self.L)
            angular_velocity = (self.velocity*np.cos(slip_angle)*np.tan(self.steering))/ self.L
        else:
            angular_velocity = 0
            slip_angle = 0

        self.angle += angular_velocity * dt
        self.position[0] += self.velocity*np.cos(self.angle+slip_angle) * dt 
        self.position[1] += self.velocity*np.sin(self.angle+slip_angle) * dt
        
        self.heading = self.angle 
        self.position_rear = [-self.Lr*np.cos(self.angle)+self.position[0], -self.Lr*np.sin(self.angle)+self.position[1]]
        self.position_front = [self.Lf*np.cos(self.angle)+self.position[0], self.Lf*np.sin(self.angle)+self.position[1]]



#* Dynamic model
    def model(self,z,t,V,delta):
        y = z[0]
        beta = z[1]
        psi = z[2]
        psidot = z[3]

        alpha_f = beta + (self.Lf/V)*psidot - delta
        alpha_r = beta - (self.Lr/V)*psidot

        if np.abs(alpha_f) <= 0.25:
            Ff = -self.C*alpha_f
        else:
            Ff = -350*np.sign(alpha_f)
        if np.abs(alpha_r) <= 0.25:
            Fr = -self.C*alpha_r
        else:
            Fr = -350*np.sign(alpha_f)

        dydt = V*beta
        dbetadt = 2/(self.mass*V)*Ff + 2/(self.mass*V)*Fr - psidot 
        dpsidt = psidot
        dpsidotdt = (2*self.Lf/self.inertia)*Ff - (2*self.Lr/self.inertia)*Fr    

        dzdt = [dydt,
                dbetadt,
                dpsidt,
                dpsidotdt]
        return dzdt

    def update_d(self,dt):
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))
        self.delta = self.steering 

        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))
        
        self.velocity += self.acceleration * dt
        self.velocity = max(0, min(self.velocity, self.max_velocity))

        # initial conditions
        z0 = [  0,  # self.position[1],
                self.beta,
                self.heading,
                self.psidot]
        # solve ODE
        # span for next time step
        tspan = [0,dt]
        # solve for next step
        z = odeint(self.model,z0,tspan,args=(self.velocity,self.delta,))
        # self.position[1] += z[1][0]
        self.beta = z[1][1]
        self.psi = z[1][2]
        self.psidot = z[1][3]
        # print(z[1])
        self.heading = self.psi

        self.position[0] += self.velocity*np.cos(self.heading+self.beta)*dt
        self.position[1] += self.velocity*np.sin(self.heading+self.beta)*dt
        
        self.position_rear = [-self.Lr*np.cos(self.heading)+self.position[0], -self.Lr*np.sin(self.heading)+self.position[1]]
        self.position_front = [self.Lf*np.cos(self.heading)+self.position[0], self.Lf*np.sin(self.heading)+self.position[1]]

