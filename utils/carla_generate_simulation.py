import json
import math
import queue

import numpy as np
import carla
import random
import os
import sys
import signal


def handle_exit(signum, frame):
    print(f"\nReceived signal {signum}, cleaning up...")
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        actorManager.clear_all()

    except Exception as e:
        print(f"Failed to reset world settings: {e}")
    sys.exit(0)  # 正常退出


class ActorManager:
    def __init__(self):
        # 设置车辆蓝图
        self.empty_bp = world.get_blueprint_library().find('util.actor.empty')
        self.dodge_bp = world.get_blueprint_library().find('vehicle.dodge.charger_2020')
        self.audi_bp = world.get_blueprint_library().find('vehicle.audi.tt')
        self.lexus_bp = world.get_blueprint_library().find('vehicle.lexus.custom')
        self.seat_bp = world.get_blueprint_library().find('vehicle.seat.leon')
        self.lincoln_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        self.bmw_bp = world.get_blueprint_library().find('vehicle.bmw.grandtourer')
        self.citroen_bp = world.get_blueprint_library().find('vehicle.citroen.c3')
        self.bus_bp = world.get_blueprint_library().find('vehicle.mitsubishi.fusorosa')
        self.truck_bp = world.get_blueprint_library().find('vehicle.carlamotors.carlacola')

        # 设置行人蓝图
        self.pedestrian_bp = world.get_blueprint_library().find('walker.pedestrian.0002')

        self.dodge_bp.set_attribute('color', '221, 50, 13')
        self.seat_bp.set_attribute('color', '247, 247, 247')

        # Actor管理
        self.actors = {}  # {actor: {flags...}}

    def spawn_actor(self, blueprint, transform):
        actor = world.spawn_actor(blueprint, transform)
        self.add_actor(actor)
        for _ in range(15):  # 等待15帧
            world.tick()
        return actor

    def add_actor(self, actor):
        """添加actor并设置初始标志位"""
        if actor not in self.actors:
            self.actors[actor] = {
                'is_camera_shot_flagged': False,
                'is_target_task_done': False,
            }

    def remove_actor(self, actor):
        """移除actor"""
        if actor in self.actors:
            actor.destroy()
            del self.actors[actor]

    def clear_all(self):
        """清除所有actor"""
        for actor in self.actors.keys():
            actor.destroy()
        self.actors.clear()

    def get_flag(self, actor, flag):
        if actor in self.actors:
            return self.actors[actor][flag]

    def set_flag(self, actor, flag, value):
        if actor in self.actors:
            self.actors[actor][flag] = value


class CameraMatrix:
    def __init__(self):
        # 设置相机属性(fov, resolution)
        self.camera_list = []
        self.depth_camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
        self.rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

        self.depth_camera_bp.set_attribute('image_size_x', str(image_size_x))
        self.depth_camera_bp.set_attribute('image_size_y', str(image_size_y))
        self.depth_camera_bp.set_attribute('fov', '120')

        self.rgb_camera_bp.set_attribute('image_size_x', str(image_size_x))
        self.rgb_camera_bp.set_attribute('image_size_y', str(image_size_y))
        self.rgb_camera_bp.set_attribute('sensor_tick', '0.1')
        self.rgb_camera_bp.set_attribute('fov', '120')

        self.rgb_camera_bp.set_attribute('exposure_mode', 'histogram')
        self.rgb_camera_bp.set_attribute('motion_blur_intensity', '0.0')
        self.rgb_camera_bp.set_attribute('blur_radius', '0.0')

        # 存储数据
        self.data_queue = queue.Queue()

    @staticmethod
    def build_projection_matrix(w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    @staticmethod
    def get_image_point(loc, K, w2c):
        point = np.array([loc.x, loc.y, loc.z, 1])
        point_camera = np.dot(w2c, point)
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
        point_img = np.dot(K, point_camera)
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    @staticmethod
    def back_project_2d_to_3d(point, depth, K, camera_transform):
        # 计算K的逆矩阵
        K_inv = np.linalg.inv(K)
        point[0] *= depth
        point[1] *= depth
        point.append(depth)

        # 反投影到相机坐标系
        point_camera = np.dot(K_inv, point)
        point_camera = [point_camera[2], point_camera[0], -point_camera[1]]

        # 转换回世界坐标
        point_world = camera_transform.transform(
            carla.Location(x=point_camera[0], y=point_camera[1], z=point_camera[2]))

        return point_world

    @staticmethod
    def is_vehicle_occluded(camera_sensor=carla.Actor, target_actor=carla.Actor):
        camera_location = camera_sensor.get_transform().location
        target_location = target_actor.get_transform().location

        if target_actor.bounding_box.contains(camera_location, target_actor.get_transform()):
            return True

        ray_results = world.cast_ray(camera_location, target_location)
        ray_labels = list(map(lambda x: x.label, ray_results))

        if target_actor.type_id == 'vehicle.carlamotors.carlacola':
            occluded_list = [carla.CityObjectLabel.Bicycle, carla.CityObjectLabel.Static, carla.CityObjectLabel.Vegetation,
                             carla.CityObjectLabel.Buildings, carla.CityObjectLabel.Fences, carla.CityObjectLabel.Poles,
                             carla.CityObjectLabel.Dynamic, carla.CityObjectLabel.Bus]
        elif target_actor.type_id == 'vehicle.mitsubishi.fusorosa':
            occluded_list = [carla.CityObjectLabel.Bicycle, carla.CityObjectLabel.Static, carla.CityObjectLabel.Vegetation,
                             carla.CityObjectLabel.Buildings, carla.CityObjectLabel.Fences, carla.CityObjectLabel.Poles,
                             carla.CityObjectLabel.Dynamic, carla.CityObjectLabel.Truck]
        else:
            occluded_list = [carla.CityObjectLabel.Bicycle, carla.CityObjectLabel.Static, carla.CityObjectLabel.Vegetation,
                             carla.CityObjectLabel.Buildings, carla.CityObjectLabel.Fences, carla.CityObjectLabel.Poles,
                             carla.CityObjectLabel.Dynamic, carla.CityObjectLabel.Truck, carla.CityObjectLabel.Bus]

        try:
            if ray_labels[0] in occluded_list:
                return True
            elif carla.CityObjectLabel.Car == ray_labels[0]:
                car_distance = target_location.distance(ray_results[0].location)
                if car_distance > 3:
                    return True
            return False
        except IndexError:
            return False


    @staticmethod
    def get_camera_transforms(distance, height, camera_number, vehicle_transform=carla.Transform):
        # 相机角度架设
        d2r_const = math.pi / 180
        one_step = round(360 / camera_number, 1)
        transforms_list = []
        for i in range(camera_number):
            # 获取角度和偏航角
            degree = i * one_step
            yaw = vehicle_transform.rotation.yaw
            # 单位转换
            rad_degree = degree * d2r_const

            # 计算offset
            offset_x = math.sin(rad_degree) * distance
            offset_y = math.cos(rad_degree) * distance

            # 相对坐标
            degree_location = carla.Location(x=offset_x, y=offset_y, z=height)
            degree_transform = carla.Transform(vehicle_transform.transform(degree_location))

            # 计算 Pitch（垂直俯仰角）
            delta_location = degree_transform.location - vehicle_transform.location
            dx, dy, dz = delta_location.x, delta_location.y, delta_location.z
            distance_2d = math.sqrt(dx ** 2 + dy ** 2)
            pitch = math.degrees(math.atan2(dz, distance_2d))

            # 设置rotation参数
            degree_transform.rotation = carla.Rotation(yaw=yaw - 90 - degree, pitch=-pitch)
            transforms_list.append((degree_transform, degree))

        return transforms_list

    # 创建相机阵列
    def spawn_camera_matrix(self, distance, height, camera_number, vehicle_transform=carla.Transform):
        # 获取camera transforms
        camera_transforms_list = self.get_camera_transforms(distance, height, camera_number, vehicle_transform)

        # 生成RGB和Depth相机
        self.camera_list = []
        for camera_transform, camera_degree in camera_transforms_list:
            self.camera_list.append((actorManager.spawn_actor(self.rgb_camera_bp, camera_transform), camera_degree))

    # 相机阵列监听
    def listen_camera_matrix(self, target_actor):
        camera_queues = {}
        for camera_actor, camera_degree in self.camera_list:
            q = queue.Queue()
            camera_queues[camera_actor] = q
            camera_actor.listen(lambda data, q=q, cam=camera_actor, deg=camera_degree:
                                q.put((data, cam, deg)))

        world.tick()

        for camera_actor, _ in self.camera_list:
            try:
                data_tuple = camera_queues[camera_actor].get(timeout=2.0)
                sensor_data, cam_actor, cam_degree = data_tuple

                self.process_sensor_data(sensor_data, cam_actor, target_actor, cam_degree)

            except queue.Empty:
                print(f"[Warning] Camera {camera_actor.id} timed out.")
                continue

        # 清理
        for cam_actor, _ in self.camera_list:
            if cam_actor.is_alive:
                cam_actor.stop()
            actorManager.remove_actor(cam_actor)
        actorManager.remove_actor(target_actor)
        self.camera_list.clear()

    # 相机阵列监听背景和前景
    def listen_background_foreground(self, distance, height, camera_number, target_actor):
        # 前景
        self.spawn_camera_matrix(distance, height, camera_number, target_actor.get_transform())
        self.listen_camera_matrix(target_actor)

    @staticmethod
    def process_sensor_data(sensor_data, sensor_actor, target_actor, sensor_degree):
        global weather_now, spawn_point_index_now

        try:
            if CameraMatrix.is_vehicle_occluded(camera_sensor=sensor_actor, target_actor=target_actor):
                # print(f"Camera {sensor_degree} occluded, skipping.")
                return
        except Exception as e:
            print(f"Occlusion check failed: {e}")
            return

        sensor_location = sensor_actor.get_transform().location
        target_location = target_actor.get_transform().location
        delta_location = target_location - sensor_location

        dist = carla.Location.distance(sensor_location, target_location)
        horizontal_dist = math.sqrt(delta_location.x ** 2 + delta_location.y ** 2)
        elev_rad = -math.atan2(delta_location.z, horizontal_dist)
        azim_rad = math.radians(sensor_degree)

        key = '{}_{}_p{}_{}'.format(TOWN_NAME, weather_now, spawn_point_index_now, int(sensor_degree))
        position_parameters[key] = {
            'dist': dist,
            'elev': elev_rad,
            'azim': azim_rad,
        }

        path = './{}/{}_{}_p{}_{}.png'.format(
            os.path.join(PARENTDIR, SUBDIR),
            TOWN_NAME,
            weather_now,
            spawn_point_index_now,
            int(sensor_degree)
        )
        sensor_data.save_to_disk(path, carla.ColorConverter.Raw)


def generate_exp_params(points_number):
    simulation_parameters = {}

    if points_number is None:
        points_number = len(spawn_points_list)

    indexed_list = list(enumerate(spawn_points_list))
    random.shuffle(indexed_list)

    for weather_name in WEATHER_PARAMETERS.keys():
        weather_params = {}
        for spawn_point_i, _ in indexed_list[:points_number]:
            weather_params[str(spawn_point_i)] = {
                'distance': random.randint(3, 15),
                'height': random.choice([1.2, 1.3, 1.4, 1.5, 1.6]),
                'camera_number': random.randint(6, 14),
            }
        simulation_parameters[weather_name] = weather_params

    os.makedirs(PARENTDIR, exist_ok=True)
    with open(os.path.join(PARENTDIR, 'exp_params.json'), "w+", encoding='utf-8') as f:
        json.dump(simulation_parameters, f, indent=4)


def generate_simulation(blueprint, params_path):
    global spawn_point_index_now, weather_now

    params_file = os.path.join(params_path, 'exp_params.json')
    with open(params_file, "r", encoding='utf-8') as f:
        all_weather_params = json.load(f)

    for weather in WEATHER_PARAMETERS:
        weather_now = weather
        world.set_weather(WEATHER_PARAMETERS[weather])

        exp_parameters = all_weather_params[weather]

        for spawn_id, params in exp_parameters.items():
            spawn_point_index_now = spawn_id
            spawn_point_transform = spawn_points_list[int(spawn_id)]
            distance = params['distance']
            height = params['height']
            camera_number = params['camera_number']

            vehicle_actor = actorManager.spawn_actor(blueprint, spawn_point_transform)

            spectator = world.get_spectator()
            spectator_transform = carla.Transform(
                spawn_point_transform.location + carla.Location(z=2),
                spawn_point_transform.rotation
            )
            spectator.set_transform(spectator_transform)

            camera_matrix.listen_background_foreground(
                distance=distance,
                height=height,
                camera_number=camera_number,
                target_actor=vehicle_actor
            )

    output_path = os.path.join(params_path, 'position_params.json')
    os.makedirs(params_path, exist_ok=True)
    with open(output_path, "w+", encoding='utf-8') as f:
        json.dump(position_parameters, f, indent=4)


if __name__ == '__main__':
    # 连接CARLA服务端
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    # 获取WORLD
    world = client.get_world()

    # 生成点
    spawn_points_list = world.get_map().get_spawn_points()  # list

    # 生成仿真数据集
    TOWN_NAME = 'town10'
    foggy_weather = carla.WeatherParameters.WetCloudyNoon
    foggy_weather.fog_density = 40.0
    foggy_weather.fog_distance = 50.0

    cloudy_weather = carla.WeatherParameters.CloudySunset
    cloudy_weather.cloudiness = 80.0

    rainy_weather = carla.WeatherParameters.MidRainSunset
    rainy_weather.cloudiness = 80.0

    WEATHER_PARAMETERS = {
        'sunny': carla.WeatherParameters.ClearNoon,
        'cloudy': cloudy_weather,
        'rainy': rainy_weather,
        'foggy': foggy_weather,
        'night': carla.WeatherParameters.ClearNight
    }

    # 仿真参数设置
    PARENTDIR = '../simulation/patch'
    SUBDIR = 'camou_citroen'
    spawn_point_index_now = None
    weather_now = None
    position_parameters = {}
    image_size_y = 320
    image_size_x = 1024

    # 实例化车辆管理类与相机管理类
    camera_matrix = CameraMatrix()
    actorManager = ActorManager()

    try:
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        settings = world.get_settings()
        settings.synchronous_mode = True  # 开启同步模式
        settings.substepping = True
        settings.fixed_delta_seconds = 0.20  # 设置固定的时间步长
        settings.max_substeps = 10
        settings.max_substep_delta_time = 0.020
        world.apply_settings(settings)

        # generate_exp_params(points_number=5)
        generate_simulation(blueprint=actorManager.citroen_bp, params_path=PARENTDIR)
        # print(world.get_blueprint_library())

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
