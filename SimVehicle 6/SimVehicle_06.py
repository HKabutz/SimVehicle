# 2019 08 21
from Vehicle_06 import vehicle
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

# Driving specifications
# X-direction path
path_x = np.round(np.arange(0,250,3.0),8)
# Y-direction path
# IMPROVE METHOD FOR DLC STEP FUNCTION
DLC_shift = 35
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
path_x = noise(path_x,0.01)
path_y = noise(path_y,0.1)
path_v = noise(path_v,0.01)
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
    def __init__(self,set_vel=0,set_accel=0,set_steer=0.0,long_cont=" ",lat_cont=" "):
        self.freq = 25          # Frequency of calculations
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

    def run(self):
        car = vehicle(1.0, -3.0, -0.4)
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

        # initial condition

        time = 0

        car.velocity = self.vel
        car.acceleration = self.accel
        car.steering = np.radians(self.steer)

        printing = False   # True
        while time < 40:
            time += self.dt

            # if time>2:
            #     car.steering = np.radians(20)
            #     if time > 3:
            #         car.steering = np.radians(-20)
            #         if time > 4:
            #            car.steering = np.radians(0.0)
            #         if time > 2.7:
            #             car.steering = np.radians(0.0)
            #             if time > 3.5:
            #                 car.steering = np.radians(1.0)

            if self.long_controller == "PD":  # TODO Longitudinal controller: Use this one!
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

            if self.lateral_controller == "PureP":  # TODO Lateral controller: Pure Pursuit
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
                # debug
                # print(path_x[i_min], path_y[i_min], goal_d, alpha, car.heading)

            elif self.lateral_controller == "FixedTan":   # TODO Lateral controller: Simple tangental drive
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

            elif self.lateral_controller == "Stanley":  # TODO Lateral Controller: Stanley front wheel steering control
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
                # print(np.degrees(steer))
                # debug
                # print(path_x[i_min], path_y[i_min], goal_d, alpha, car.heading)

            # TODO (just for colour) Lateral Controller: Path curvature Model predict     1      2019-07-12
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
                g = error_look  # TODO cross track error based steering function
                h = 0  # TODO rate of change of heading angle - inertia based term
                steer = 1*f + 0*g + 0*h + psi*1
                if steer > np.pi / 2:
                    steer = steer - np.pi
                steer = max(-0.35, min(steer, 0.35))
                car.set_steering(steer, self.dt)
                # print(car.steering,"  psi: ",psi)

            # TODO (just for colour) Lateral Controller: Path curvature Model predict     2     2019-07-13
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
                g = error_look  # TODO cross track error based steering function
                h = 0  # TODO rate of change of heading angle - inertia based term
                steer = 1.5*f + 1.5*f2 + 0*g + 0*h + psi*1
                if steer > np.pi / 2:
                    steer = steer - np.pi
                steer = max(-0.35, min(steer, 0.35))
                car.set_steering(steer, self.dt)
                # print(car.steering,"  psi: ",psi)

            # TODO (just for colour) Lateral Controller: Path curvature Model predict     3     2019-07-13
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
                g = errorlook       # TODO cross track error based steering function
                h = 0               # TODO rate of change of heading angle - inertia based term

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


            # Vehicle physics model
            car.update_d(self.dt)


            if printing == True:
                print("Time: ",np.round(time,5)," Freq: ",self.dt," position: ",car.position," V: ",car.velocity,
                        " A: ",car.acceleration," Steering: ",car.steering," Slip angle: ",car.beta)
                if self.lateral_controller == "Pure":
                    print("Steer: ",np.degrees(car.steering))
                if self.long_controller == "P":
                    print("v_des: ",p_desired_v)

            # Draw line after vehicle
            p_history_x.append(car.position[0])
            p_history_y.append(car.position[1])
            p_history_steer.append(np.degrees(car.steering))


        self.temp_hist_index = np.array(hist_index) * self.vel * self.dt
        self.temp_hist_error = hist_error
        self.temp_hist_error_x = hist_error_x
        self.temp_p_hist_x = np.asarray(p_history_x)
        self.temp_p_hist_y = np.asarray(p_history_y)
        self.temp_p_hist_steer = np.asarray(p_history_steer)

        self.temp_track_K = hist_track_K
        self.temp_track_psi = hist_track_psi
        self.temp_track_g = hist_track_g
        

if __name__ == '__main__':
    sim1 = Simulator(set_vel=10, long_cont=" ",lat_cont="CurvatureFollow3")
    sim1.run()

    sim2 = Simulator(set_vel=5, long_cont=" ",lat_cont="CurvatureFollow2")
    sim2.run()

    # sim15 = simulator(set_vel=15, long_cont=" ",lat_cont="FixedTan")
    # sim15.run()

    # sim20 = simulator(set_vel=20, long_cont=" ",lat_cont="FixedTan")
    # sim20.run()

    # plt.figure(2)
    # plt.plot(sim5.temp_hist_index, sim5.temp_hist_error,"m-",label='At 5m/s',linewidth=2)
    # plt.plot(sim10.temp_hist_index, sim10.temp_hist_error,"y-",label='At 10m/s',linewidth=2)
    # # plt.plot(sim15.temp_hist_index, sim15.temp_hist_error,"r-",label='At 15m/s',linewidth=2) 
    # # plt.plot(sim20.temp_hist_index, sim20.temp_hist_error,"g-",label='At 20m/s',linewidth=2)
    # plt.xlabel("Distance traveled [m]")
    # plt.ylabel("Error [m]")
    # plt.axis([0,200,0,4])
    # plt.title('Cross track error')
    # plt.legend(loc='upper right')


    # plt.figure(1)
    # plt.plot(path_x , path_y,"b-",label='Desired Path',linewidth=3)
    # plt.plot(sim5.temp_p_hist_x, sim5.temp_p_hist_y,"m-",label='Actual Path at 5m/s',linewidth=2)
    # plt.plot(sim10.temp_p_hist_x, sim10.temp_p_hist_y,"y-",label='Actual Path at 10m/s',linewidth=2)
    # # plt.plot(sim15.temp_p_hist_x, sim15.temp_p_hist_y,"r-",label='Actual Path at 15m/s',linewidth=2)
    # # plt.plot(sim20.temp_p_hist_x, sim20.temp_p_hist_y,"g-",label='Actual Path at 20m/s',linewidth=2)
    # plt.xlabel("Horizontal position [m]")
    # plt.ylabel("Lateral position [m]")
    # plt.title('Tangental Drive Position Controller')
    # plt.axis([0,200,-5,5])#'scaled')#
    # plt.legend(loc='lower right')

    plt.figure(1)
    plt.subplot(311)
    plt.plot(path_x, path_y,"b-",label='Desired Path',linewidth=3)
    # plt.plot(path_x, path_a, "r-", label='Desired Path gradient', linewidth=3)
    # plt.plot(path_x, 10 * path_k, "g.", label='Desired Path curvature', linewidth=3)
    plt.plot(sim1.temp_p_hist_x, sim1.temp_p_hist_y,"m-",label='CurvatureFollow3 at ##m/s',linewidth=2)
    # plt.plot(sim2.temp_p_hist_x, sim2.temp_p_hist_y,"r-",label='Actual Path at 18m/s CurvatureFollow',linewidth=2)
    # plt.plot(sim15.temp_p_hist_x, sim15.temp_p_hist_y,"r-",label='Actual Path at 15m/s',linewidth=2)
    # plt.plot(sim20.temp_p_hist_x, sim20.temp_p_hist_y,"g-",label='Actual Path at 20m/s',linewidth=2)
    plt.xlabel("Horizontal position [m]")
    plt.ylabel("Lateral position [m]")
    plt.title('Double Lange Change Maneuver')
    plt.axis([0,200,-20,20])#'scaled')#
    plt.legend(loc='lower right')

    plt.subplot(312)
    plt.plot(sim1.temp_p_hist_x, sim1.temp_hist_error,"m-",label='CurvatureFollow3 at ##m/s',linewidth=2)
    # plt.plot(sim2.temp_p_hist_x, sim2.temp_hist_error,"r-",label='CurvatureFollow2 at 18m/s',linewidth=2)
    # plt.plot(sim15.temp_hist_index, sim15.temp_hist_error,"r-",label='At 15m/s',linewidth=2) 
    # plt.plot(sim20.temp_hist_index, sim20.temp_hist_error,"g-",label='At 20m/s',linewidth=2)
    plt.xlabel("Distance traveled [m]")
    plt.ylabel("Error [m]")
    plt.axis([0,200,0,3])
    plt.title('Cross track error')
    plt.legend(loc='upper right')

    plt.subplot(313)
    plt.plot((0,200),(17,17),"g--")
    plt.plot((0,200),(-17,-17), "g--")
    plt.plot(sim1.temp_p_hist_x, sim1.temp_p_hist_steer,"m-",label='CurvatureFollow3 at ##m/s')
    #  plt.plot(sim2.temp_p_hist_x, sim2.temp_p_hist_steer,"r-",label='CurvatureFollow2 at 18m/s')
    plt.xlabel("Distance traveled [m]")
    plt.ylabel("Steering angle in degrees")
    plt.title('Steering angle')
    plt.axis([0,200,-20,20])
    plt.legend(loc='upper right')

    plt.tight_layout(pad=0.0)

    plt.figure(2)
    plt.plot(sim1.temp_p_hist_x, sim1.temp_track_K, "y-", label='K')
    plt.plot(sim1.temp_p_hist_x, sim1.temp_track_psi, "b-", label='psi')
    plt.plot(sim1.temp_p_hist_x, sim1.temp_track_g, "g-", label='g')
    plt.xlabel("Distance traveled [m]")
    plt.ylabel("Weight scale")
    plt.title('Controller weights')
    plt.xlim(0,200)
    plt.legend(loc='upper right')



    plt.show()



