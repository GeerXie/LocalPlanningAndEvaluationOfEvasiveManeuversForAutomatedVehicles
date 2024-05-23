import numpy as np
import matplotlib.pyplot as plt
import math

class PIDController:
    def __init__(self, kp, ki, kd, target=0):
        self.kp = kp  
        self.ki = ki  
        self.kd = kd  
        self.target = target 
        self.integral = 0
        self.previous_error = 0

    def update(self, error):
        print(error,"error")
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output


class TrajectoryPlanner:
    def __init__(self, s0, v0, a0, sf, vf, af, ttc, ego_v):
        self.s0 = s0
        self.v0 = v0
        self.a0 = a0 / 2 
        self.sf = sf
        self.vf = vf
        self.af = af
        self.tf = ttc
        self.ego_v = ego_v

        T = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 2, 0, 0, 0],
                      [1, self.tf, self.tf**2, self.tf**3, self.tf**4, self.tf**5],
                      [0, 1, 2*self.tf, 3*self.tf**2, 4*self.tf**3, 5*self.tf**4],
                      [0, 0, 2, 6*self.tf, 12*self.tf**2, 20*self.tf**3]])
        Q = [self.s0, self.v0, self.a0, self.sf, self.vf, self.af]
        self.coefficients = np.linalg.solve(T, Q)

    def calc_x_coord(self, t):
        x_cal = self.ego_v * t
        return x_cal

    def calc_y_coord(self, t):
        c0, c1, c2, c3, c4, c5 = self.coefficients
        y_cal = c0 + c1 * t + c2 * t **2 + c3 * t**3 + c4 * t**4 + c5 * t**5
        return  y_cal


class Vehicle:
    def __init__(self, x, y, delta, vel, dt):
        self.vel = vel
        self.x = x
        self.y = y
        self.delta = delta
        self.dt = dt
        self.beta = 0 
        self.r = 0  
        self.yaw = 0  
        self.pid_controller = PIDController(kp=175, ki=53, kd=100)
        self.init_vehicle_properties()

    def init_vehicle_properties(self):
        self.length = 2.338
        self.width = 1.381
        self.rear_to_wheel = 0.339
        self.wheel_length = 0.531
        self.wheel_width = 0.125
        self.track = 1.094
        self.wheel_base = 1.686
        self.Caf = 2*32857.5
        self.Car = 2*32857.5
        self.mass = 633
        self.lf = 0.9442
        self.lr = 0.7417
        self.Iz = 430.166

    def motion_model(self, y_cal):
        current_position = self.y
        error = y_cal - current_position
        # whether y_cal and self.y is at the same t ime domin?
        steering_adjustment = self.pid_controller.update(error)
        self.delta += steering_adjustment
        self.delta = self.delta
        print(self.delta)

        def deg_to_rad(deg):
            return deg * math.pi / 180
        
        coefficient_A = np.array([
            [-1*(self.Caf + self.Car) / (self.mass * self.vel), ((self.Car * self.lr - self.Caf * self.lf) / (self.mass * self.vel**2)) - 1],
            [(self.Car * self.lr - self.Caf * self.lf) / self.Iz, -(self.Caf * self.lf**2 + self.Car * self.lr**2) / (self.Iz * self.vel)]
        ])
        
        state_vector = np.array([[self.beta], [self.r]])
        coefficient_B = np.array([[self.Caf / (self.mass * self.vel)], [self.Caf * self.lf / self.Iz]])
        control_matrix = np.array([[self.delta]])
        
        output_matrix = np.dot(coefficient_A, state_vector) + np.dot(coefficient_B, control_matrix)
        beta_dot = output_matrix[0][0]
        r_dot = output_matrix[1][0]

        self.beta += beta_dot * self.dt
        self.r += r_dot * self.dt
        self.yaw += self.r * self.dt

        if self.yaw > 180:
            self.yaw -= 360
        elif self.yaw < -180:
            self.yaw += 360

        x_dot = self.vel * math.cos(deg_to_rad(self.yaw + self.beta))
        y_dot = self.vel * math.sin(deg_to_rad(self.yaw + self.beta))
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt
        
        return self.x, self.y


def calculate_trajectory(s0, v0, a0, sf, vf, af, ego_v, ttc, ped_v=None, time=None):
    trajectory = TrajectoryPlanner(s0, v0, a0, sf, vf, af, ttc, ego_v)
    t_list = np.arange(0, ttc, 0.01)
    vehicle_x_list = [trajectory.calc_x_coord(t) + (time if time else 0) for t in t_list]
    vehicle_y_list = [trajectory.calc_y_coord(t) for t in t_list]
    result = [(vehicle_x_list, vehicle_y_list, "Trajectory")]
    if ped_v is not None:
        pedestrian_x_start = 6
        pedestrian_y_start = -2
        pedestrian_x_list = [time + pedestrian_x_start] * len(t_list)
        pedestrian_y_list = [pedestrian_y_start + ped_v * t for t in t_list]
        index_to_trim = next((i for i, y in enumerate(pedestrian_y_list) if y >= 0), None)
        pedestrian_x_list = pedestrian_x_list[:index_to_trim]
        pedestrian_y_list = pedestrian_y_list[:index_to_trim]
        result.append((pedestrian_x_list, pedestrian_y_list, "Pedestrian Trajectory"))
    return result



def main():
    s0, v0, a0, sf, vf, af, ego_v, ped_v = 0, 0, 0, 0, 0, 0, 8.3, 3
    
    traj1 = calculate_trajectory(s0, v0, a0, sf, vf, af, ego_v, 0.5)
    s0 = traj1[0][1][-1]
    sf = 3.5

    time = traj1[0][0][-1]
    
    traj2 = calculate_trajectory(s0, v0, a0, sf, vf, af, ego_v, 6/ego_v, ped_v, time)
    s0 = traj2[0][1][-1]
    time = traj2[0][0][-1]
    sf = 3.5
    
    traj3 = calculate_trajectory(s0, v0, a0, sf, vf, af, ego_v, 0.1, None, time)
    s0 = traj3[0][1][-1]
    sf = 0
    time = traj3[0][0][-1]
    
    traj4 = calculate_trajectory(s0, v0, a0, sf, vf, af, ego_v, 0.8, None, time)

    vehicle = Vehicle(traj2[0][0][0], traj2[0][1][0], 0, ego_v, 0.01)
    vehicle_trajectory_x = []
    vehicle_trajectory_y = []

    for y in traj2[0][1]:
        vehicle.motion_model(y)
        
        vehicle_trajectory_x.append(vehicle.x)
        vehicle_trajectory_y.append(vehicle.y)


    # for x, y in zip(traj2[0][0][0], traj2[0][1][0]):
    #     print(traj2[0][1],"traj2[0][1]")
    #     vehicle_trajectory_x.append(vehicle.x)
    #     vehicle_trajectory_y.append(vehicle.y)
    

    plot_trajectories([traj1, traj2, traj3, traj4], (vehicle_trajectory_x, vehicle_trajectory_y))

def plot_trajectories(trajectories, vehicle_trajectory=None):
    plt.figure(figsize=(10, 6))
    plt.title("Trajectories")

    for trajectory_group in trajectories:
        for trajectory in trajectory_group:
            vehicle_x, vehicle_y, label = trajectory
            plt.plot(vehicle_x, vehicle_y, label=label)

    if vehicle_trajectory:

        plt.plot(vehicle_trajectory[0], vehicle_trajectory[1], label="Vehicle Trajectory", color='red')

    #if pedestrian_trajectory:
        #plt.scatter(*pedestrian_trajectory, label="Pedestrian Trajectory", color='red', s=10)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
