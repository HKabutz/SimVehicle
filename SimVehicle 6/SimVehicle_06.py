# 2019 08 21
from Vehicle_06 import vehicle
import numpy as np
import scipy.stats as sps
import scipy.optimize as opt
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
path_v = np.round(5 + path_x*0, 3)
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

# Gradient
def gradient(point_a, point_b):
    x = point_a[0] - point_b[0]
    y = point_a[1] - point_b[1]
    return np.arctan2(y, x)


def path_gradient(index_1, index_2):
    return gradient((path_x[index_2], path_y[index_2]), (path_x[index_1], path_y[index_1]))

# path_a - Path gradient function in angle of path to horizontal
path_len = len(path_x)
path_a = [path_gradient(0, 1)]
for i in range(1, path_len - 1):
    path_a.append(path_gradient(i-1, i+1))
path_a.append(path_gradient(path_len - 2, path_len - 1))
path_a = np.array(path_a)


# Curvature
def curvature(x1, y1, x2, y2, x3, y3):
    # (-x1*(y3-y2) + x2*(y3-y1) - x3*(y2-y1)) / ((((x3-x2)**2+(x2-x1)**2)/2 + ((y3-y2)**2+(y2-y1)**2)/2)**(3/2))
    return 8*(-x1*(y3-y2) + x2*(y3-y1) - x3*(y2-y1)) / (((x3-x1)**2 + (y3-y1)**2)**(3/2))


def path_curvature(i_1, i_2, i_3):
    return curvature(path_x[i_1], path_y[i_1], path_x[i_2], path_y[i_2], path_x[i_3], path_y[i_3])


# path_k - Path curvature
path_k = [0,0]
for i in range(2, path_len - 2):
    path_k.append(path_curvature(i-2, i, i+2))
path_k.append(0)  # path_curvature(path_len - 3,path_len-2 , path_len - 1))
path_k.append(0)
path_k = np.array(path_k)





# plt.figure()
# #plt.plot(path_x, path_a, "r-", label='Desired Path gradient', linewidth=3)
# plt.plot(path_x, path_y, "b-", label='Desired Path', linewidth=3)
# plt.plot(path_x, 10*path_k, "g.", label='Desired Path curvature', linewidth=3)
# plt.axis([0,200,-15,15])
# plt.legend(loc='lower right')
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
        opt_f = opt_par[0]
        opt_psi = opt_par[1]
        opt_g = opt_par[2]

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

        diff_hist = [0, 0]

        self.MPC_t_x1 = []
        MPC_t_x2 = []
        self.MPC_t_y1 = []
        MPC_t_y2 = []

        # initial condition

        time = 0

        car.velocity = self.vel
        car.acceleration = self.accel
        car.steering = np.radians(self.steer)

        while time < 25:
            time += self.dt
            #  print("time: ",time )

            x_noise_temp = sps.norm.rvs(loc=0, scale=0.1)
            y_noise_temp = sps.norm.rvs(loc=0, scale=0.1)
            H_noise_temp = sps.norm.rvs(loc=0, scale=0.6)


            if self.long_controller == "PD":  # Longitudinal controller: Use this one!
                Kp = 15
                Kd = 8
                
                p_error = np.sqrt((path_x - car.position[0])**2 + (path_y - car.position[1])**2)
                index_p_error_min = np.argmin(p_error)
                p_desired_v = path_v[index_p_error_min]

                p_hist.append(path_x[index_p_error_min])
                v_history_path.append(p_desired_v)
                v_history_x.append(car.velocity)
                v_diff.append(p_desired_v-car.velocity)

                a = Kp*(v_diff[len(v_diff)-1]) + Kd*(v_diff[len(v_diff)-1]-v_diff[len(v_diff)-3])/(2*self.dt) 
                # Kd*(car.velocity - v_history_x[len(v_history_x)-3])/(2*self.dt)
                car.set_acceleration(a,self.dt)

            elif self.long_controller == "PID":  # Dont use
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

            if self.lateral_controller == "PureP":  # Lateral controller: Pure Pursuit
                # Vehicle Position (rear point)
                x_veh = car.position_rear[0]
                y_veh = car.position_rear[1]
                # Lookahead distance (meters)
                look = 11
                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff**2 + y_diff**2)
                # Error is given by distance between vehicle and closest point on path to vehicle
                error_pos = np.min(r_diff)
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                hist_error.append(r_diff[error_pos-1])
                hist_error_x.append(x_veh)
                # Difference between look and r_diff. Gives the path point that is look distance from vehicle
                r_look = r_diff - look
                r_look = np.abs(r_look)
                # index of the point on path
                i_min = np.argmin(r_look)
                # goal point, x, y and distance from vehicle
                goal_x = path_x[i_min]
                goal_y = path_y[i_min]
                goal_d = r_diff[i_min]
                # angle of straight line path from vehicle to goal point
                angle_direct_path = np.arctan2(goal_y - y_veh, goal_x - x_veh)
                # angle between vehicle heading and goal point
                alpha = angle_direct_path - car.heading
                alpha = (alpha+np.pi) % (2*np.pi) - np.pi
                # turning radius to achieve arc to goal point
                rho = goal_d / (2 * np.sin(alpha)+0.000000000000001)
                # steering angle to turning radius
                steer = np.arctan2(car.L, rho)
                if steer >np.pi/2:
                    steer = steer - np.pi
                car.set_steering(steer,self.dt)
                print(np.degrees(steer))
                # print(path_x[i_min], path_y[i_min], goal_d, alpha, car.heading)

            elif self.lateral_controller == "FixedTan":   #  Lateral controller: Simple tangental drive
                # Vehicle Position (rear point)
                x_veh = car.position_rear[0] + x_noise_temp
                y_veh = car.position_rear[1] + y_noise_temp
                # Lookahead r_rels and angle
                look_min = 0.1  # 5  # 0.1
                look_max = 3  # 20
                look_angle = 1.4
                # Difference (r_rel) between vehicle and path points
                x_rel = path_x - x_veh
                y_rel = path_y - y_veh
                # Distance and angle between vehicle and path points
                r_rel = np.sqrt(x_rel ** 2 + y_rel ** 2)
                g_rel = wrap(car.heading + H_noise_temp - np.arctan2(y_rel, x_rel))
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

                hist_track_K.append(steer)
                hist_track_psi.append(0)
                hist_track_g.append(0)

            elif self.lateral_controller == "PID_follow":   #  Lateral controller: Simple tangental drive
                # Vehicle Position (rear point)
                x_veh = car.position_rear[0] + x_noise_temp
                y_veh = car.position_rear[1] + y_noise_temp
                # Lookahead r_rels and angle
                look_min = 0.1  # 5  # 0.1
                look_max = 3  # 20
                look_angle = 1.4
                # Difference (r_rel) between vehicle and path points
                x_rel = path_x - x_veh
                y_rel = path_y - y_veh
                # Distance and angle between vehicle and path points
                r_rel = np.sqrt(x_rel ** 2 + y_rel ** 2)
                # HISTORY STUFF
                hist_index.append(len(hist_index))
                hist_error.append(y_rel[np.argmin(r_rel)])
                hist_error_x.append(x_veh)

                y_rel_min = y_rel[np.argmin(r_rel)]

                # TODO Calculating the heading from previous results is not viable, as the derivative is critical for the heading.
                # using the heading of the vehicle based on the current and previous position, a slight error in location
                # has a large error in heading. Thus becoming very inaccurate.
                Kp = 0.1
                Kd = 0.1
                Kdh = -0.1
                Ki = 0.0

                diff = y_rel_min
                diff_hist.append(diff)
                integrate = np.sum(np.array(diff_hist)) * self.dt
                derivative = (diff_hist[-1] - diff_hist[-2]) / self.dt
                pid_heading = car.heading + H_noise_temp

                steer = Kp * diff + Kd * derivative + Ki * integrate + Kdh * pid_heading

                if steer >np.pi/2:
                    steer = steer - np.pi
                car.set_steering(steer,self.dt)
                # print(steer)

                hist_track_K.append(Kp * diff)
                hist_track_psi.append(Kd * derivative)
                hist_track_g.append(Kdh * pid_heading)

            elif self.lateral_controller == "Stanley":  # Lateral Controller: Stanley front wheel steering control
                # Vehicle Position (front point)
                x_veh = car.position_front[0]
                y_veh = car.position_front[1]
                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff**2 + y_diff**2)
                # Error is given by distance between vehicle and closest point on path to vehicle
                error_pos = r_diff.argmin()
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                hist_error.append(r_diff[error_pos-1])
                hist_error_x.append(x_veh)
                angle = path_a[error_pos]
                psi = angle - car.heading

                k_stanley_1 = 10
                k_stanley_2 = 1000
                steer = psi + np.arctan2(k_stanley_1 * r_diff[error_pos], k_stanley_2 + car.velocity)
                if steer > np.pi/2:
                    steer = steer - np.pi
                car.set_steering(steer, self.dt)

                hist_track_K.append(steer)
                hist_track_psi.append(0)
                hist_track_g.append(0)
                # print(np.degrees(steer))

            # Lateral Controller: Path curvature Model predict     1      2019-07-12
            elif self.lateral_controller == "CurvatureFollow":
                # vehicle properties
                M = 173
                V = car.velocity
                C = 1400
                L = 0.72    # m    Wheel base length
                Lf = 0.35   # m    CG to front axis length
                Lr = L - Lf
                look = 0.3 * car.velocity

                # Vehicle Position (front point)
                x_veh = car.position[0]
                y_veh = car.position[1]
                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)
                # Error is given by distance between vehicle and closest point on path to vehicle
                error_pos = r_diff.argmin()
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                hist_error.append(r_diff[error_pos - 1])
                hist_error_x.append(x_veh)

                # Calculate look ahead point error and position
                x_veh_look = look * np.cos(car.heading) + x_veh
                y_veh_look = look * np.sin(car.heading) + y_veh
                x_diff_look = path_x - x_veh_look
                y_diff_look = path_y - y_veh_look
                r_diff_look = np.sqrt(x_diff_look ** 2 + y_diff_look ** 2)
                error_pos_look = r_diff_look.argmin()
                error_look = r_diff_look[error_pos_look]
                K = path_k[error_pos_look]

                angle = path_a[error_pos_look]

                psi = angle - car.heading  # Angle between the vehicle heading and the path tangental

                f = K * (L+(Lr-Lf)*(M*V**2)/(2*C*L))  # dynamic steering function for curvature
                g = error_look  # cross track error based steering function
                h = 0  # rate of change of heading angle - inertia based term
                steer = 1*f + 0*g + 0*h + psi*1
                if steer > np.pi / 2:
                    steer = steer - np.pi
                steer = max(-0.35, min(steer, 0.35))
                car.set_steering(steer, self.dt)
                # print(car.steering,"  psi: ",psi)

            # Lateral Controller: Path curvature Model predict     2     2019-07-13
            elif self.lateral_controller == "CurvatureFollow2":
                # vehicle properties
                M = 173
                V = car.velocity
                C = 1400
                L = 0.72    # m    Wheel base length
                Lf = 0.35   # m    CG to front axis length
                Lr = L - Lf
                look = 0.3 * car.velocity
                look2 = 0.15 * car.velocity

                # Vehicle Position (front point)
                x_veh = car.position[0]
                y_veh = car.position[1]
                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)
                # Error is given by distance between vehicle and closest point on path to vehicle
                error_pos = r_diff.argmin()
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                # hist_error.append(r_diff[error_pos])
                hist_error.append(y_diff[error_pos])  # only the y error is shown,
                hist_error_x.append(x_veh)
                # but the overall error is used for calculations


                # Calculate look ahead point error and position
                x_veh_look = look * np.cos(car.heading) + x_veh
                y_veh_look = look * np.sin(car.heading) + y_veh
                x_diff_look = path_x - x_veh_look
                y_diff_look = path_y - y_veh_look
                r_diff_look = np.sqrt(x_diff_look ** 2 + y_diff_look ** 2)
                error_pos_look = r_diff_look.argmin()
                error_look = r_diff_look[error_pos_look]

                K = path_k[error_pos_look]
                angle = path_a[error_pos_look]

                # Calculate look ahead point error and position
                x_veh_look2 = look2 * np.cos(car.heading) + x_veh
                y_veh_look2 = look2 * np.sin(car.heading) + y_veh
                x_diff_look2 = path_x - x_veh_look2
                y_diff_look2 = path_y - y_veh_look2
                r_diff_look2 = np.sqrt(x_diff_look2 ** 2 + y_diff_look2 ** 2)
                error_pos_look2 = r_diff_look2.argmin()

                K2 = path_k[error_pos_look2]

                psi = angle - car.heading  #  Angle between the vehicle heading and the path tangental
                f = K * (L+(Lr-Lf)*(M*V**2)/(2*C*L))  # dynamic steering function for curvature
                f2 = K2 * (L + (Lr - Lf) * (M * V ** 2) / (2 * C * L))
                g = error_look  # cross track error based steering function
                h = 0  # rate of change of heading angle - inertia based term
                steer = 1.5*f + 1.5*f2 + 0*g + 0*h + psi*1
                if steer > np.pi / 2:
                    steer = steer - np.pi
                steer = max(-0.35, min(steer, 0.35))
                car.set_steering(steer, self.dt)
                # print(car.steering,"  psi: ",psi)

            # Lateral Controller: Path curvature Model predict     3     2019-07-13
            elif self.lateral_controller == "CurvatureFollow3":
                # vehicle properties
                M = 173
                V = car.velocity
                C = 1400
                L = 0.72  # m    Wheel base length
                Lf = 0.35  # m    CG to front axis length
                Lr = L - Lf

                lookClose = 0.1 * car.velocity
                lookFar = 0.8 * car.velocity
                lookRanges = 8  # Number of devisions in the look distance
                lookChange = (lookFar-lookClose)/lookRanges
                lookPoints = []
                for ilook in range(lookRanges+1):
                    lookPoints.append(lookClose + ilook*lookChange)
                lookPoints = np.array(lookPoints)


                # Vehicle Position (front point)
                x_veh = car.position[0]
                y_veh = car.position[1]
                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)
                # Error is given by location of distance between vehicle and closest point on path to vehicle
                error_pos = r_diff.argmin()
                error_veh = np.abs(y_diff[error_pos])
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                # hist_error.append(r_diff[error_pos])
                hist_error.append(error_veh)  # only the y error is shown,
                hist_error_x.append(x_veh)
                # but the overall error is used for calculations

                # Calculate look ahead point error and position
                lookPErrorPos = []
                lookPError = []
                lookPK = []
                lookPAngle = []
                for i in range(len(lookPoints)):
                    x_veh_look = lookPoints[i] * np.cos(car.heading) + x_veh
                    y_veh_look = lookPoints[i] * np.sin(car.heading) + y_veh
                    x_diff_look = path_x - x_veh_look
                    y_diff_look = path_y - y_veh_look
                    r_diff_look = np.sqrt(x_diff_look ** 2 + y_diff_look ** 2)
                    error_pos_look = r_diff_look.argmin()
                    error_look = np.sign(y_diff_look[error_pos_look])*r_diff_look[error_pos_look]


                    KAtPoint = path_k[error_pos_look]
                    angleAtPoint = path_a[error_pos_look]

                    lookPErrorPos.append(error_pos_look)
                    lookPError.append(error_look)
                    lookPK.append(KAtPoint)
                    lookPAngle.append(angleAtPoint)

                #  sensitivity fit the controller results on the distance
                lookPK = np.array(lookPK)
                lookPError = np.array(lookPError)
                lookPAngle = np.array(lookPAngle)

                K = np.sum(lookPK * sensitivity(hCenter=lookRanges/2,hScale=1,hOffset=0,size=lookRanges))
                angle =     np.sum(lookPAngle * sensitivity(hCenter=lookRanges/3,hScale=1,hOffset=0,size=lookRanges))
                errorlook = np.sum(lookPError * sensitivity(hCenter=lookRanges/3,hScale=1,hOffset=0,size=lookRanges))

                f = K * (L + (Lr - Lf) * (M * V ** 2) / (2 * C * L))  # dynamic steering function for curvature
                psi = angle - car.heading  # Angle between the vehicle heading and the path tangental
                g = errorlook       # cross track error based steering function
                h = 0               # rate of change of heading angle - inertia based term

                #  print('f: ',f,'   psi: ',psi, '   g: ',g)

                f = 3.0 * f
                psi = 0.6 * psi
                g = 0.02 * g

                f = max(-0.35, min(f, 0.35))
                psi = max(-0.3, min(psi, 0.3))
                g = max(-0.3, min(g, 0.3))

                steer = f + psi + g
                if steer > np.pi / 2:
                    steer = steer - np.pi
                steer = max(-0.3, min(steer, 0.3))
                car.set_steering(steer, self.dt)

                hist_track_K.append(f)
                hist_track_psi.append(psi)
                hist_track_g.append(g)
                # print(car.steering,"  psi: ",psi)

            elif self.lateral_controller == "MPC_simple":
                # vehicle properties
                M = 173
                V = car.velocity
                C = 1400
                L = 0.72  # m    Wheel base length
                Lf = 0.35  # m    CG to front axis length
                Lr = L - Lf

                MPC_dt = 0.3
                MPC_horizon = 15

                # Vehicle Position (front point)
                x_veh = car.position[0]
                y_veh = car.position[1]
                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)
                # Error is given by location of distance between vehicle and closest point on path to vehicle
                error_pos = r_diff.argmin()
                error_veh = np.abs(y_diff[error_pos])
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                # hist_error.append(r_diff[error_pos])
                hist_error.append(error_veh)  # only the y error is shown,
                hist_error_x.append(x_veh)


                MPC_x = x_veh
                MPC_y = y_veh

                def MPC_opt(a):
                    opt_J = 0
                    opt_a = 0
                    mpc_angle = car.heading
                    mpc_position_rear_x = -Lr*np.cos(car.heading)+MPC_x
                    mpc_position_rear_y = -Lr*np.sin(car.heading)+MPC_y
                    aS = car.steering
                    for i in range(len(a)):
                        aS += a[i]
                        aS = max(-0.3, min(aS, 0.3))
                        if aS:
                            turning_radius = L / np.tan(aS)
                            angular_velocity = V / turning_radius
                        else:
                            angular_velocity = 0

                        mpc_angle += angular_velocity * MPC_dt
                        mpc_position_rear_x += V * np.cos(mpc_angle) * MPC_dt
                        mpc_position_rear_y += V * np.sin(mpc_angle) * MPC_dt

                        MPC_X = Lr * np.cos(mpc_angle) + mpc_position_rear_x
                        MPC_Y = Lr * np.sin(mpc_angle) + mpc_position_rear_y

                        MPC_x_diff = path_x - MPC_X
                        MPC_y_diff = path_y - MPC_Y
                        MPC_r_diff = np.sqrt(MPC_x_diff ** 2 + MPC_y_diff ** 2)
                        MPC_r_location = MPC_r_diff.argmin()
                        opt_J += np.abs(MPC_y_diff[MPC_r_location])
                        opt_a += np.abs(path_a[MPC_r_location]-mpc_angle)
                        opt = opt_J + opt_a*4
                    return opt


                initial = np.zeros(MPC_horizon)

                MPC_steer_opt = opt.minimize(MPC_opt,initial,constraints=({'type':'ineq','fun':lambda a:a+0.1},{'type':'ineq','fun':lambda a:0.1-a}))
                print(MPC_steer_opt.x)
                MPC_steer = MPC_steer_opt.x[0]
                print(MPC_steer+car.steering)
                car.set_steering(car.steering+MPC_steer, self.dt)

            elif self.lateral_controller == "MPC_SS":
                # vehicle properties
                M = 173
                V = car.velocity
                C = 1400
                L = 0.72  # m    Wheel base length
                Lf = 0.35  # m    CG to front axis length
                Lr = L - Lf

                MPC_dt = 0.4
                MPC_horizon = 10

                # Vehicle Position (front point)
                x_veh = car.position[0]
                y_veh = car.position[1]
                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)
                # Error is given by location of distance between vehicle and closest point on path to vehicle
                error_pos = r_diff.argmin()
                error_veh = np.abs(y_diff[error_pos])
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                # hist_error.append(r_diff[error_pos])
                hist_error.append(error_veh)  # only the y error is shown,
                hist_error_x.append(x_veh)


                MPC_x = x_veh
                MPC_y = y_veh

                MPC_t_x2 = []
                MPC_t_y2 = []

                def MPC_opt(a):
                    opt_J = 0
                    opt_a = 0
                    mpc_angle = car.heading
                    mpc_position_rear_x = -Lr*np.cos(car.heading)+MPC_x
                    mpc_position_rear_y = -Lr*np.sin(car.heading)+MPC_y
                    aS = car.steering
                    for i in range(len(a)):
                        aS += a[i]
                        aS = max(-0.3, min(aS, 0.3))
                        R = 1/(aS + 0.00000000000001) * (L+(Lr-Lf)*(M*V**2)/(2*C*L))
                        angular_velocity = V/R
                        mpc_angle_current = angular_velocity * MPC_dt

                        dX = R*np.sin(mpc_angle_current)
                        dY = R*(1-np.cos(mpc_angle_current))

                        mpc_position_rear_x += np.cos(mpc_angle)*dX - np.sin(mpc_angle)*dY
                        mpc_position_rear_y += np.sin(mpc_angle)*dX + np.cos(mpc_angle)*dY

                        mpc_angle += mpc_angle_current

                        MPC_X = Lr * np.cos(mpc_angle) + mpc_position_rear_x
                        MPC_Y = Lr * np.sin(mpc_angle) + mpc_position_rear_y

                        MPC_t_x2.append(MPC_X)
                        MPC_t_y2.append(MPC_Y)

                        MPC_x_diff = path_x - MPC_X
                        MPC_y_diff = path_y - MPC_Y
                        MPC_r_diff = np.sqrt(MPC_x_diff ** 2 + MPC_y_diff ** 2)
                        MPC_r_location = MPC_r_diff.argmin()
                        opt_J += np.abs(MPC_y_diff[MPC_r_location]**2)
                        opt_a += np.abs((path_a[MPC_r_location]-mpc_angle))
                        opt = opt_J + opt_a*20
                    return opt

                self.MPC_t_x1.append(MPC_t_x2)
                self.MPC_t_y1.append(MPC_t_y2)

                initial = np.zeros(MPC_horizon)

                MPC_steer_opt = opt.minimize(MPC_opt,initial,constraints=({'type':'ineq','fun':lambda a:a+0.1},{'type':'ineq','fun':lambda a:0.1-a}))
                print(MPC_steer_opt.x)
                MPC_steer = MPC_steer_opt.x[0]
                print(MPC_steer+car.steering)
                car.set_steering(car.steering+MPC_steer, self.dt)

            # Lateral Controller: Path curvature Model predict     4     2019-08-01
            elif self.lateral_controller == "Opt_CFollow1":
                # vehicle properties
                M = 173
                V = car.velocity
                C = 1400
                L = 0.72  # m    Wheel base length
                Lf = 0.35  # m    CG to front axis length
                Lr = L - Lf

                lookClose = 0.1 * car.velocity
                lookFar = 0.8 * car.velocity  # TODO Change the look ahead distance
                lookRanges = 8  # Number of devisions in the look distance
                lookChange = (lookFar - lookClose) / lookRanges
                lookPoints = []
                for ilook in range(lookRanges + 1):
                    lookPoints.append(lookClose + ilook * lookChange)
                lookPoints = np.array(lookPoints)

                # Vehicle Position (front point)
                x_veh = car.position[0] + x_noise_temp
                y_veh = car.position[1] + y_noise_temp

                # Difference (distance) between vehicle and path points
                x_diff = path_x - x_veh
                y_diff = path_y - y_veh
                r_diff = np.sqrt(x_diff ** 2 + y_diff ** 2)
                # Error is given by location of distance between vehicle and closest point on path to vehicle
                error_pos = r_diff.argmin()
                error_veh = y_diff[error_pos]  # deleted: np.abs()
                # History of error (for graphs)
                hist_index.append(len(hist_index))
                # hist_error.append(r_diff[error_pos])
                hist_error.append(error_veh)  # only the y error is shown,
                hist_error_x.append(x_veh)
                # but the overall error is used for calculations

                # Calculate look ahead point error and position
                lookPErrorPos = []
                lookPError = []
                lookPK = []
                lookPAngle = []
                for i in range(len(lookPoints)):
                    x_veh_look = lookPoints[i] * np.cos(car.heading + H_noise_temp) + x_veh
                    y_veh_look = lookPoints[i] * np.sin(car.heading + H_noise_temp) + y_veh
                    x_diff_look = path_x - x_veh_look
                    y_diff_look = path_y - y_veh_look
                    r_diff_look = np.sqrt(x_diff_look ** 2 + y_diff_look ** 2)
                    error_pos_look = r_diff_look.argmin()
                    error_look = np.sign(y_diff_look[error_pos_look]) * r_diff_look[error_pos_look]

                    KAtPoint = path_k[error_pos_look]
                    angleAtPoint = path_a[error_pos_look]

                    lookPErrorPos.append(error_pos_look)
                    lookPError.append(error_look)
                    lookPK.append(KAtPoint)
                    lookPAngle.append(angleAtPoint)

                #  sensitivity fit the controller results on the distance
                lookPK = np.array(lookPK)
                lookPError = np.array(lookPError)
                lookPAngle = np.array(lookPAngle)

                K = np.sum(lookPK * sensitivity(hCenter=lookRanges / 2, hScale=1, hOffset=0, size=lookRanges))
                angle = np.sum(lookPAngle * sensitivity(hCenter=lookRanges / 3, hScale=1, hOffset=0, size=lookRanges))
                errorlook = np.sum(lookPError * sensitivity(hCenter=lookRanges / 3, hScale=1, hOffset=0, size=lookRanges))

                f = K * (L + (Lr - Lf) * (M * V ** 2) / (2 * C * L))  # dynamic steering function for curvature
                psi = angle - car.heading - H_noise_temp  # Angle between the vehicle heading and the path tangental
                g = errorlook  # cross track error based steering function
                h = 0  # TODO rate of change of heading angle - inertia based term

                #  print('f: ',f,'   psi: ',psi, '   g: ',g)

                opt_f = opt_par[0]
                opt_psi = opt_par[1]

                f = opt_f * f       # 3.0 * f            TODO Imporve the feed-forward effect
                psi = opt_psi * psi    # 0.6 * psi       TODO Make more stable for errors in point position
                g = opt_g * g       # 0.02 * g           TODO Add integral term effect

                f = max(-0.3, min(f, 0.3))
                psi = max(-0.3, min(psi, 0.3))
                g = max(-0.3, min(g, 0.3))

                steer = f + psi + g
                if steer > np.pi / 2:
                    steer = steer - np.pi
                steer = max(-0.3, min(steer, 0.3))
                car.set_steering(steer, self.dt)

                hist_track_K.append(f)
                hist_track_psi.append(psi)
                hist_track_g.append(g)
                # print(car.steering,"  psi: ",psi)

            # Vehicle physics model
            car.update_d(self.dt)



            # Draw line after vehicle
            p_history_x.append(car.position[0]+x_noise_temp)
            p_history_y.append(car.position[1]+y_noise_temp)
            p_history_steer.append(np.degrees(car.steering))


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


def simOptimize(opt_A):
    sum_opt_error = []
    vvv = np.array([7.5])
    yyy = np.array([-1,0,1])
    aaa = np.array([0,0.1])

    # vvv = np.array([2, 5, 7.5, 10, 12.5, 15])
    # yyy = np.array([-4, -2, -1, 0, 1, 2, 4])
    # aaa = np.array([-0.2, -0.1, 0, 0.1, 0.2])

    for VV in vvv:
        for YY in yyy:
            for AA in aaa:
                simOpt = Simulator(set_vel=VV, long_cont=" ", lat_cont="Opt_CFollow1")
                simOpt.setProperties(xx=1.0, yy=YY, aa=AA, set_vel=VV)
                simOptError = simOpt.run(opt_par=opt_A)
                print("X ", 1.0, "   Y ", YY, "   a ", AA, "   vel ", VV,"    param ",opt_A,"     Error ",simOptError)
                sum_opt_error.append(simOptError)

    sum_opt_error = np.sum(np.fabs(np.array(sum_opt_error)))
    return sum_opt_error


if __name__ == '__main__':
    initial = [3.0,0.6,0.06]
    #Opt_CFollow = opt.minimize(simOptimize, initial, constraints=({'type': 'ineq', 'fun': lambda opt_A: opt_A}, {'type': 'ineq', 'fun': lambda opt_A: 5-opt_A}))
    #print(Opt_CFollow)

    sim1 = Simulator(set_vel=0, long_cont=" ", lat_cont="PID_follow")  # "Opt_CFollow1"   "MPC_simple"   "FixedTan"  "PID_follow"
    sim1.setProperties(xx=1.0, yy=1.0, aa=-0.30, set_vel=2.00)
    # sim1.run(opt_par=Opt_CFollow.x)
    # sim1.run(opt_par=[0.42296772, 0.36308098, 0.11522529])
    sim1.run(opt_par=[0, 0.6, 0.2])


    plt.figure(1)
    plt.subplot(311)
    plt.plot(path_x, path_y,"b.",label='Desired Path',linewidth=3)
    plt.plot(sim1.temp_p_hist_x, sim1.temp_p_hist_y,"m-",label=sim1.lateral_controller+' at '+str(sim1.vel)+'m/s',linewidth=2)
    #  plt.plot(sim1.MPC_t_x1,sim1.MPC_t_y1)
    plt.xlabel("Horizontal position [m]")
    plt.ylabel("Lateral position [m]")
    plt.title('Double Lange Change Maneuver')
    plt.axis([0,160,-5,5])#'scaled')#
    plt.legend(loc='lower right')

    plt.subplot(312)
    plt.plot((0,160),(0,0),"g--")
    plt.plot(sim1.temp_p_hist_x, sim1.temp_hist_error,"m-",label=sim1.lateral_controller+' at '+str(sim1.vel)+'m/s',linewidth=2)
    plt.xlabel("Distance traveled [m]")
    plt.ylabel("Error [m]")
    plt.axis([0,160,-2,2])
    plt.title('Cross track error')
    plt.legend(loc='upper right')

    plt.subplot(313)
    plt.plot((0,160),(17,17),"g--")
    plt.plot((0,160),(-17,-17), "g--")
    plt.plot(sim1.temp_p_hist_x, sim1.temp_track_K, "y-", label='K')
    plt.plot(sim1.temp_p_hist_x, sim1.temp_track_psi, "b-", label='psi')
    plt.plot(sim1.temp_p_hist_x, sim1.temp_track_g, "g-", label='g')
    plt.plot(sim1.temp_p_hist_x, sim1.temp_p_hist_steer,"m-",label=sim1.lateral_controller+' at '+str(sim1.vel)+'m/s',linewidth=3)
    plt.xlabel("Distance traveled [m]")
    plt.ylabel("Steering angle in degrees")
    plt.title('Steering angle')
    plt.axis([0,160,-20,20])
    plt.legend(loc='upper right')

    plt.tight_layout(pad=0.0)




    plt.show()



