import json
import math
import numpy as np
import carla
import random
import os
import signal
import sys
import queue


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
        self.dodge_bp.set_attribute('color', '221, 50, 13')

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
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # Convert from UE4's coordinate system to a "standard" system (x, y, z) -> (y, -z, x)
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
        # Project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # Normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    @staticmethod
    def back_project_2d_to_3d(point, depth, K, camera_transform):
        # 计算K的逆矩阵
        K_inv = np.linalg.inv(K)

        # reverse coordinate
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
    def calculate_relative_sun_direction(mode="UE5", **kwargs):
        if mode.upper() == "UE5":
            params = dict(
                vehicle_yaw_degrees=0.0,
                time_of_day=1200.0,
                sun_pitch=0.0,
                sun_yaw=0.0,
                sun_vertical_offset=0.0,
            )
            params.update(kwargs)

            day_progress = (params["time_of_day"] - 600) / 1200.0
            base_azimuth_deg = (day_progress * 180.0) - 90.0 + 180.0

            final_azimuth_deg = base_azimuth_deg + params["sun_yaw"]
            azimuth_rad = np.radians(final_azimuth_deg)

            path_angle_rad = day_progress * np.pi
            max_elevation_deg = 90.0 - params["sun_pitch"]
            current_elevation_deg = np.sin(path_angle_rad) * max_elevation_deg
            final_elevation_deg = current_elevation_deg + params["sun_vertical_offset"]
            elevation_rad = np.radians(final_elevation_deg)

            x_pos = np.cos(elevation_rad) * np.cos(azimuth_rad)
            y_pos = np.sin(elevation_rad)
            z_pos = np.cos(elevation_rad) * np.sin(azimuth_rad)
            sun_direction_world = np.array([x_pos, y_pos, z_pos])
            vehicle_rot_rad = np.radians(params["vehicle_yaw_degrees"])

            cos_a = np.cos(vehicle_rot_rad)
            sin_a = np.sin(vehicle_rot_rad)
            rotation_matrix_y = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            final_light_direction = rotation_matrix_y @ sun_direction_world
            return list(final_light_direction)

        elif mode.upper() == "UE4":
            params = dict(
                vehicle_yaw_degrees=0.0,
                sun_azimuth_angle=0.0,
                sun_altitude_angle=45.0,
            )
            params.update(kwargs)
            azimuth_rad = np.radians(params["sun_azimuth_angle"])
            altitude_rad = np.radians(params["sun_altitude_angle"])

            x = np.cos(altitude_rad) * np.cos(azimuth_rad)
            y = np.sin(altitude_rad)
            z = np.cos(altitude_rad) * np.sin(azimuth_rad)

            sun_dir = np.array([x, y, z], dtype=np.float32)

            vehicle_rot_rad = np.radians(-params["vehicle_yaw_degrees"])
            cos_a, sin_a = np.cos(vehicle_rot_rad), np.sin(vehicle_rot_rad)
            rotation_matrix_y = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            return list(rotation_matrix_y @ sun_dir)

        else:
            raise ValueError("mode must be 'UE5' or 'UE4'")

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
                # 获取数据
                data_tuple = camera_queues[camera_actor].get(timeout=2.0)
                sensor_data, cam_actor, cam_degree = data_tuple

                self.process_sensor_data(sensor_data, cam_actor, target_actor, cam_degree)

            except queue.Empty:
                print(f"[Warning] Camera {camera_actor.id} timed out waiting for data")
                continue

        actorManager.set_flag(target_actor, 'is_target_task_done', True)
        for cam_actor, _ in self.camera_list:
            if cam_actor.is_alive:
                cam_actor.stop()
            actorManager.remove_actor(cam_actor)
        actorManager.remove_actor(target_actor)
        self.camera_list.clear()

    # 相机阵列监听背景和前景
    def listen_background_foreground(self, distance, height, camera_number, target_actor, empty_actor):
        # 前景
        self.spawn_camera_matrix(distance, height, camera_number, target_actor.get_transform())
        self.listen_camera_matrix(target_actor)

        # 背景
        self.spawn_camera_matrix(distance, height, camera_number, empty_actor.get_transform())
        self.listen_camera_matrix(empty_actor)


    # 相机传回数据
    @staticmethod
    def process_sensor_data(sensor_data, sensor_actor, target_actor, sensor_degree):
        global weather_now, frame_count, frame_count_empty, position_parameters

        try:
            if CameraMatrix.is_vehicle_occluded(camera_sensor=sensor_actor, target_actor=target_actor):
                return
        except Exception as e:
            print(f"Occlusion check failed: {e}")
            return

        if target_actor.type_id == 'util.actor.empty':
            path = '{}/{}_p{}_{}_{}.png'.format(
                os.path.join(PARENTDIR, SUBDIR, 'background'),
                TOWN_NAME, spawn_point_index, weather_now['name'], frame_count_empty
            )
            sensor_data.save_to_disk(path, carla.ColorConverter.Raw)
            frame_count_empty += 1

        else:
            sensor_location = sensor_actor.get_transform().location
            target_location = target_actor.get_transform().location
            delta_location = target_location - sensor_location

            dist = carla.Location.distance(sensor_location, target_location)
            horizontal_dist = math.sqrt(delta_location.x ** 2 + delta_location.y ** 2)
            elev_rad = -math.atan2(delta_location.z, horizontal_dist)
            azim_rad = math.radians(sensor_degree)

            # 恢复完整的参数列表
            position_data = {
                'dist': dist,
                'elev': elev_rad,
                'azim': azim_rad,
                'light': CameraMatrix.calculate_relative_sun_direction(
                    mode='UE4',
                    vehicle_yaw_degrees=target_actor.get_transform().rotation.yaw,
                    sun_azimuth_angle=weather_now['param'].sun_azimuth_angle,
                    sun_altitude_angle=weather_now['param'].sun_altitude_angle)
            }

            key = f"{TOWN_NAME}_p{spawn_point_index}_{weather_now['name']}_{frame_count}"
            position_parameters[key] = position_data

            path = '{}/{}_p{}_{}_{}.png'.format(
                os.path.join(PARENTDIR, SUBDIR, 'rgb'),
                TOWN_NAME, spawn_point_index, weather_now['name'], frame_count
            )
            sensor_data.save_to_disk(path, carla.ColorConverter.Raw)
            frame_count += 1

# 生成数据集
def generate_dataset(vehicle_bp, is_continue: bool, is_shuffle: bool, points_number=None):
    global frame_count, frame_count_empty, position_parameters, spawn_point_index, weather_now

    # 预取的生成点数量
    if points_number is None:
        points_number = len(spawn_points_list)

    # 是否为连续生成数据集
    if is_continue:
        try:
            files = os.listdir(os.path.join(PARENTDIR, SUBDIR, 'rgb'))
            frame_count = len(files)
            frame_count_empty = len(files)
            with open(os.path.join(PARENTDIR, SUBDIR, 'positions.json'), "r", encoding='utf-8') as f:
                position_parameters = json.load(f)
        except FileNotFoundError:
            frame_count = 0
            frame_count_empty = 0
    else:
        frame_count = 0
        frame_count_empty = 0

    # 生成点列表
    indexed_list = list(enumerate(spawn_points_list))
    if is_shuffle:
        random.shuffle(indexed_list)
    for weather_name, weather_param in WEATHER_PARAMETERS.items():
        weather_now.update({'name': weather_name})
        original_azimuth = weather_param.sun_azimuth_angle
        original_altitude = weather_param.sun_altitude_angle
        for i, transform in indexed_list[:points_number]:
            '''
            distance : 3 ~ 15m, camera_numbers:4 ~ 12  ,height: 1.1m ~ 1.5m
            '''
            spawn_point_index = i
            random_distance = random.randint(3, 15)
            random_height = random.choice([1.2, 1.3, 1.4, 1.5, 1.6])
            random_camera_number = random.randint(6, 14)

            weather_param.sun_azimuth_angle = original_azimuth + random.uniform(-2.0, 2.0)
            weather_param.sun_altitude_angle = original_altitude + random.uniform(-2.0, 2.0)
            weather_now.update({'param': weather_param})
            world.set_weather(weather_now['param'])

            vehicle_transform = transform
            vehicle_actor = actorManager.spawn_actor(vehicle_bp, vehicle_transform)
            empty_actor = actorManager.spawn_actor(actorManager.empty_bp, vehicle_actor.get_transform())

            # 调整CARLA观察者视角
            spectator = world.get_spectator()
            spectator_transform = carla.Transform(vehicle_transform.location + carla.Location(z=2), vehicle_transform.rotation)
            spectator.set_transform(spectator_transform)

            # 生成相机阵列并收集数据
            camera_matrix.listen_background_foreground(distance=random_distance, height=random_height,
                                                       camera_number=random_camera_number, target_actor=vehicle_actor,
                                                       empty_actor=empty_actor)

    # 输出位姿参数文件
    with open(os.path.join(PARENTDIR, SUBDIR, 'positions.json'), "w+", encoding='utf-8') as f:
        json.dump(position_parameters, f, indent=4)


if __name__ == '__main__':
    # 连接CARLA服务端
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)

    # 获取WORLD
    world = client.get_world()

    # 生成点
    spawn_points_list = world.get_map().get_spawn_points()  # list

    # 生成数据集
    position_parameters = {}
    frame_count = None
    frame_count_empty = None
    spawn_point_index = None
    weather_now = {'name': 'sunny', 'param': world.get_weather()}
    image_size_y = 320
    image_size_x = 1024

    # 仿真数据集参数设置
    PARENTDIR = '../dataset'
    SUBDIR = 'test'
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

    # 实例化车辆管理类、CAMERA类
    actorManager = ActorManager()
    camera_matrix = CameraMatrix()

    # 生成数据集
    try:
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        settings = world.get_settings()
        settings.synchronous_mode = True  # 开启同步模式
        settings.substepping = True
        settings.fixed_delta_seconds = 0.16  # 设置固定的时间步长
        settings.max_substeps = 10
        settings.max_substep_delta_time = 0.016
        world.apply_settings(settings)

        generate_dataset(vehicle_bp=actorManager.lexus_bp, is_continue=True, is_shuffle=False, points_number=3)

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
