# 2019 08 21
from Vehicle_06 import vehicle
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

# Driving specifications
# X-direction path
path_x = np.round(np.arange(0,250,0.20),8)
# Y-direction path
# IMPROVE METHOD FOR DLC STEP FUNCTION
DLC_shift = 20
path_y = np.round(  (0 * 0.5*(1-np.sign(path_x-15-DLC_shift))) +
                    ((3.5/2*(1-np.cos(np.pi/30*(path_x-15-DLC_shift)))) * 0.25*(1+np.sign(path_x-15-DLC_shift))*(1-np.sign(path_x-45-DLC_shift))) +
                    (3.5 * 0.25*(1+np.sign(path_x-45-DLC_shift))*(1-np.sign(path_x-70-DLC_shift))) +
                    ((3.5/2*(1+np.cos(np.pi/30*(path_x-70-DLC_shift)))) * 0.25*(1+np.sign(path_x-70-DLC_shift))*(1-np.sign(path_x-100-DLC_shift))) +
                    (0 * 0.5*(1-np.sign(path_x-100-DLC_shift))), 8)  # DLC
# path_y = np.round(-(0*(path_x<100))+(0*(path_x>100))+1*np.sin(path_x/2),8)  # sinusoidal path
# Velocity path profile
path_v = np.round(1.5 + path_x*0, 3)
# path_v = np.round(3*np.sin(path_x/17)+15,3)
# path_v = np.round(0.0*path_x+10,3)

# time_x = np.arange(0,100,0.1)
# time_v = np.round(0.0*time_x+5,3)

def noise(path_array, noise_level):
    for u in range(0,len(path_array)-1):
        path_array[u] += sps.norm.rvs(loc=0,scale=noise_level)
    return path_array

# plt.plot(path_x,path_y)
# print(path_x)

# path_x = noise(path_x,0.3)  # Adding noise to the path.
# path_y = noise(path_y,0.3)
# path_v = noise(path_v,0.01)

# print(path_x)
# plt.plot(path_x,path_y)
# plt.show()



def sensitivity(hCenter,hScale,hOffset,size):
    x = np.arange(0, size+1)
    nor = sps.norm.pdf(x, loc=hCenter, scale=size / 3)
    ynorm = nor / np.sum(nor)*hScale + hOffset
    return ynorm




def rotate(px,py,angle,ox,oy):
    pnx = np.cos(angle)*(px-ox)-np.sin(angle)*(py-oy)+ox
    pny = np.sin(angle)*(px-ox)+np.cos(angle)*(py-oy)+oy
    return pnx, pny


def wrap(x, a=-np.pi, b=np.pi):
    return (x-a) % (b-a) + a


# Simulator
class Simulator:
    def __init__(self,set_vel=0.0,set_accel=0.0,set_steer=0.0,long_cont=" ",lat_cont=" "):
        self.freq = 10          # Frequency of calculations
        self.dt = 1/self.freq   # time interval between calculations
 
        self.vel = set_vel
        self.accel = set_accel
        self.steer = set_steer

        self.temp_hist_index = []
        self.temp_hist_error = []
        self.temp_hist_error_x = []
        self.temp_p_hist_x = []
        self.temp_p_hist_y = []
        self.temp_p_hist_steer = []
        self.temp_track_K = []
        self.temp_track_psi = []
        self.temp_track_g = []

        self.long_controller = long_cont    # " " #"PD"  #"PID"
        self.lateral_controller = lat_cont  # "FixedTan" #"PureP" # "MPC"  #"Stanly" "PD

        self.xx = 1.0
        self.yy = 0.0
        self.aa = 0.0

    def setProperties(self,xx=1.0,yy=0.0,aa=0.0,set_vel=0.0,set_accel=0.0,set_steer=0.0):
        self.xx = xx
        self.yy = yy
        self.aa = aa
        self.vel = set_vel
        self.accel = set_accel
        self.steer = set_steer

    def run(self,opt_par=[0.0,0.0,0.0]):

        car = vehicle(self.xx, self.yy, self.aa)
        p_history_x = []
        p_history_y = []
        p_history_steer = []
        p_hist = []
        v_history_x = [0]
        v_history_path = []
        v_diff = [0,0]
        v_integrate = 0
        v_desire_p_hist = []
        hist_index = []
        hist_error = []
        hist_error_x = []
        hist_time = []
        hist_track_K = []
        hist_track_psi = []
        hist_track_g = []

        velll_time = []
        velll_vel = []
        velll_prev = 0
        velll_accel = []
        velll_accel2 = []
        integral = 0


        self.MPC_t_x1 = []
        MPC_t_x2 = []
        self.MPC_t_y1 = []
        MPC_t_y2 = []

        # initial condition

        time = 0

        velll_desired = 1
        car.velocity = self.vel
        car.acceleration = self.accel
        car.steering = np.radians(self.steer)


        while time < 7:
            time += self.dt
            #  print("time: ",time )

            x_noise_temp = 0 # sps.norm.rvs(loc=0, scale=0.3)
            y_noise_temp = 0 # sps.norm.rvs(loc=0, scale=0.3)
            H_noise_temp = sps.norm.rvs(loc=0, scale=0.001)



            if self.long_controller == "PI2":
                Kp = 0.2
                Ki = 0.3


                velll_time.append(time)

                velll_error = velll_desired - car.velocity # + sps.norm.rvs(loc=0, scale=1)
                velll_drive = 1 * velll_error


                p_error = np.sqrt((path_x - car.position[0]) ** 2 + (path_y - car.position[1]) ** 2)
                index_p_error_min = np.argmin(p_error)
                p_desired_v = path_v[index_p_error_min]
                p_hist.append(path_x[index_p_error_min])
                v_history_path.append(p_desired_v)
                v_history_x.append(car.velocity)
                v_diff.append(p_desired_v - car.velocity)



                max_vel_change = 1 * self.dt # m2/s
                if np.abs(velll_drive - velll_prev) > max_vel_change:
                    velll_prev = velll_prev + np.sign(velll_drive - velll_prev) * max_vel_change
                    print(velll_prev)
                else:
                    velll_prev = velll_drive

                car.velocity = velll_prev
                velll_vel.append(velll_prev)



            elif self.long_controller == "PID":  # Longitudinal controller: Use this one!
                Kp = 5
                Kd = 0.9
                Ki = 0.0

                velll_error = velll_desired - car.velocity - H_noise_temp
                velll_drive = 1 * velll_error
                
                p_error = np.sqrt((path_x - car.position[0])**2 + (path_y - car.position[1])**2)
                index_p_error_min = np.argmin(p_error)
                p_desired_v = path_v[index_p_error_min]

                p_hist.append(path_x[index_p_error_min])
                v_history_path.append(p_desired_v)
                v_history_x.append(car.velocity+H_noise_temp)

                v_diff.append(1.5-car.velocity-H_noise_temp)

                integral += v_diff[-1] * self.dt

                a = Kp*(v_diff[-1]) + Ki * integral + Kd*(v_diff[-1]-v_diff[-3])/(2*self.dt)
                # Kd*(car.velocity - v_history_x[len(v_history_x)-3])/(2*self.dt)
                car.set_acceleration(a,self.dt)
                velll_accel.append(a)

                velll_accel2.append(car.acceleration)

                velll_prev = car.velocity
                velll_vel.append(velll_prev)
                velll_time.append(time)


            elif self.long_controller == "PID3":  # Dont use
                Kp = 15
                Kd = 0
                Ki = 0
                
                p_error = np.sqrt((path_x - car.position[0])**2 + (path_y - car.position[1])**2)
                index_p_error_min = np.argmin(p_error)
                p_desired_v = path_v[index_p_error_min]

                p_hist.append(path_x[index_p_error_min])
                v_history_path.append(p_desired_v)
                v_history_x.append(car.velocity)
                v_diff.append(p_desired_v-car.velocity)
                
                v_integrate += (car.velocity + v_history_x[len(v_diff)-2])/2 * self.dt

                a = Kp*(v_diff[len(v_diff)-1]) + Kd*(v_diff[len(v_diff)-1]-v_diff[len(v_diff)-3])/(2*self.dt) + Ki*v_integrate
                car.set_acceleration(a,self.dt)

            elif self.long_controller == "PDt":  # Don't Use
                Kp = 5
                Kd = 0.2
                t_diff_pos = np.argmin(np.abs(time_v-time))
                p_desired_v = time_v[t_diff_pos]

                hist_time.append(time)
                v_path_hist.append(p_desired_v)
                v_history_x.append(car.velocity)
                v_diff.append(p_desired_v-car.velocity)

                a = Kp*(v_diff[len(v_diff)-1]) + Kd*(car.velocity - v_history_x[len(v_history_x)-2])/(self.dt) 
                # #Kd*(v_diff[len(v_diff)-1]-v_diff[len(v_diff)-3])/(2*self.dt)
                car.set_acceleration(a,self.dt)

            elif self.lateral_controller == "FixedTan":   #  Lateral controller: Simple tangental drive
                # Vehicle Position (rear point)
                x_veh = car.position_rear[0]
                y_veh = car.position_rear[1]
                # Lookahead r_rels and angle
                look_min = 0.1  # 5  # 0.1
                look_max = 3  # 20
                look_angle = 1.4
                # Difference (r_rel) between vehicle and path points
                x_rel = path_x - x_veh
                y_rel = path_y - y_veh
                # Distance and angle between vehicle and path points
                r_rel = np.sqrt(x_rel ** 2 + y_rel ** 2)
                g_rel = wrap(car.heading - np.arctan2(y_rel, x_rel))
                # HISTORY STUFF
                hist_index.append(len(hist_index))
                hist_error.append(np.min(r_rel))
                hist_error_x.append(x_veh)
                # determine the error
                delta = np.abs(path_a + 2*g_rel - car.heading)
                # point inclusion criteria
                idx = (look_min < r_rel) * (look_max > r_rel) * (np.abs(g_rel) < look_angle) * (delta < np.pi / 4)
                # valid path  point indices
                ids = np.where(idx)
                # error for valid points
                dels = delta[ids]
                # decide on goal point by minimum error
                i_goal = (0 if not len(dels) else ids[0][np.argmin(dels)])
                # turning radius to achieve arc to goal point
                rho = r_rel[i_goal] / (2 * np.sin(-g_rel[i_goal])+0.00000000001)
                # steering angle to turning radius
                steer = np.arctan2(car.L, rho)
                if steer >np.pi/2:
                    steer = steer - np.pi
                car.set_steering(steer,self.dt)
                # print(steer)


            # Vehicle physics model
            car.update_d(self.dt)



            # Draw line after vehicle
            p_history_x.append(car.position[0]+x_noise_temp)
            p_history_y.append(car.position[1]+y_noise_temp)
            p_history_steer.append(np.degrees(car.steering))

        self.temp_time = np.array(velll_time)
        self.temp_vel = np.array(velll_vel)
        self.temp_accel = np.array(velll_accel)
        self.temp_accel2 = np.array(velll_accel2)


        self.temp_hist_index = np.array(hist_index) * self.vel * self.dt
        self.temp_hist_error = hist_error
        self.temp_hist_error_x = hist_error_x
        self.temp_p_hist_x = np.asarray(p_history_x)
        self.temp_p_hist_y = np.asarray(p_history_y)
        self.temp_p_hist_steer = np.asarray(p_history_steer)

        self.temp_track_K = np.array(hist_track_K)*57.3
        self.temp_track_psi = np.array(hist_track_psi)*57.3
        self.temp_track_g = np.array(hist_track_g)*57.3
        opt_error = np.sum(np.array(hist_error))
        return opt_error


if __name__ == '__main__':
    initial = [3.0,0.6,0.06]
    # Opt_CFollow = opt.minimize(simOptimize, initial, constraints=({'type': 'ineq', 'fun': lambda opt_A: opt_A}, {'type': 'ineq', 'fun': lambda opt_A: 5-opt_A}))
    # print(Opt_CFollow)

    sim1 = Simulator(set_vel=0, long_cont="PID")
    sim1.setProperties(xx=1.0, yy=1.0, aa=-0.00, set_vel=0.00)
    # sim1.run(opt_par=Opt_CFollow.x)
    sim1.run(opt_par=[0.42296772, 0.36308098, 0.11522529])

    plt.figure(1)
    plt.subplot(211)
    plt.plot(sim1.temp_time, sim1.temp_time*0+1.5,"b--",label='Desired Path',linewidth=3)
    plt.plot(sim1.temp_time, sim1.temp_vel,"r-",label=sim1.long_controller+' ',linewidth=2)
    #  plt.plot(sim1.MPC_t_x1,sim1.MPC_t_y1)
    plt.xlabel("Time [s]")
    plt.ylabel("Vehicle Speed [m]")
    plt.title('Vehicle speed controller')
    # plt.axis([0,160,-5,5]) #'scaled')#
    plt.legend(loc='lower right')

    plt.subplot(212)
    #plt.plot(sim1.temp_time, sim1.temp_time * 0 + 1, "b--", label='Desired Path', linewidth=3)
    plt.plot(sim1.temp_time, sim1.temp_accel, "m-", label=sim1.long_controller + ' desired',linewidth=2)
    plt.plot(sim1.temp_time, sim1.temp_accel2, "g-", label=sim1.long_controller + ' actual',linewidth=2)
    #  plt.plot(sim1.MPC_t_x1,sim1.MPC_t_y1)
    plt.xlabel("Time [s]")
    plt.ylabel("Vehicle acceleration [m]")
    plt.title('Vehicle speed controller acceleration')
    # plt.axis([0,160,-5,5]) #'scaled')#
    plt.legend(loc='upper right')




    # plt.subplot(312)
    # plt.plot((0,160),(0,0),"g--")
    # plt.plot(sim1.temp_p_hist_x, sim1.temp_hist_error,"m-",label=sim1.lateral_controller+' at '+str(sim1.vel)+'m/s',linewidth=2)
    # plt.xlabel("Distance traveled [m]")
    # plt.ylabel("Error [m]")
    # plt.axis([0,160,-2,2])
    # plt.title('Cross track error')
    # plt.legend(loc='upper right')
    #
    # plt.subplot(313)
    # plt.plot((0,160),(17,17),"g--")
    # plt.plot((0,160),(-17,-17), "g--")
    # plt.plot(sim1.temp_p_hist_x, sim1.temp_track_K, "y-", label='K')
    # plt.plot(sim1.temp_p_hist_x, sim1.temp_track_psi, "b-", label='psi')
    # plt.plot(sim1.temp_p_hist_x, sim1.temp_track_g, "g-", label='g')
    # plt.plot(sim1.temp_p_hist_x, sim1.temp_p_hist_steer,"m-",label=sim1.lateral_controller+' at '+str(sim1.vel)+'m/s',linewidth=3)
    # plt.xlabel("Distance traveled [m]")
    # plt.ylabel("Steering angle in degrees")
    # plt.title('Steering angle')
    # plt.axis([0,160,-20,20])
    # plt.legend(loc='upper right')
    #
    plt.tight_layout(pad=0.50)




    plt.show()



