import carla
import time
import numpy as np
import cv2
import random
from pathlib import Path

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load Town02
client.load_world('Town02')
world = client.get_world()

# Set up the world and get the map
blueprint_library = world.get_blueprint_library()
map = world.get_map()

# Set the weather conditions
weather = carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, sun_altitude_angle=90.0)
world.set_weather(weather)

# Set up the spectator
spectator = world.get_spectator()
spectator.set_transform(carla.Transform(carla.Location(x=180, y=-140, z=50), carla.Rotation(pitch=-90)))

# Create vehicle blueprints
vehicle_bp = blueprint_library.filter("model3")[0]

# Create pedestrian blueprints
pedestrian_bp = blueprint_library.filter("walker.pedestrian.*")[0]

# Spawn vehicles and pedestrians
vehicles = []
pedestrians = []
spawn_points = map.get_spawn_points()

# Function to check if a spawn point is free of collisions
def is_spawn_point_free(spawn_point, actor_list, distance_threshold=3.0):
    for actor in actor_list:
        if actor.get_location().distance(spawn_point.location) < distance_threshold:
            return False
    return True

# Spawn vehicles at random locations
for i in range(5):
    random_spawn_point = random.choice(spawn_points)
    while not is_spawn_point_free(random_spawn_point, vehicles):
        random_spawn_point = random.choice(spawn_points)

    vehicle = world.spawn_actor(vehicle_bp, random_spawn_point)
    vehicle.set_autopilot(True)
    vehicles.append(vehicle)

#for i in range(5):
    # Find a free spawn point for the pedestrian
    #free_spawn_point = None
    #for spawn_point in spawn_points:
    #    if is_spawn_point_free(spawn_point, vehicles):
    #        free_spawn_point = spawn_point
    #        break
#
    #if free_spawn_point is not None:
    #    pedestrian = world.spawn_actor(pedestrian_bp, free_spawn_point)
    #    pedestrians.append(pedestrian)
    #else:
    #    print("Warning: Could not find a free spawn point for pedestrian", i)

# Set up RGB and semantic segmentation cameras
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '512')
camera_bp.set_attribute('image_size_y', '256')
camera_bp.set_attribute('fov', '110')

semseg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
semseg_bp.set_attribute('image_size_x', '512')
semseg_bp.set_attribute('image_size_y', '256')
semseg_bp.set_attribute('fov', '110')

# Attach cameras to the first vehicle
rgb_camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)), attach_to=vehicles[0])
semseg_camera = world.spawn_actor(semseg_bp, carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)), attach_to=vehicles[0])

# Create directories to save the frames
rgb_dir = Path("rgb_frames")
semseg_dir = Path("semseg_frames")

rgb_dir.mkdir(parents=True, exist_ok=True)
semseg_dir.mkdir(parents=True, exist_ok=True)


# Function to save the images
def save_image(image, path):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    cv2.imwrite(str(path / f"{image.frame_number:06d}.png"), array)

# Function to process semantic segmentation data
def process_semseg(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array[:, :, 2]

# Function to process and save semantic segmentation images
def save_semseg(image, path):
    array = process_semseg(image)
    cv2.imwrite(str(path / f"{image.frame_number:06d}.png"), array)


# Lists to store file paths for the saved images
rgb_image_paths = []
semseg_image_paths = []

# Register the callback functions to save images and store file paths
def save_rgb_image(image):
    save_image(image, rgb_dir)
    rgb_image_paths.append(rgb_dir / f"{image.frame_number:06d}.png")
    #print("RGB image saved:", rgb_dir / f"{image.frame_number:06d}.png")


def save_semseg_image(image):
    save_semseg(image, semseg_dir)
    semseg_image_paths.append(semseg_dir / f"{image.frame_number:06d}.png")
    #print("SemSeg image saved:", semseg_dir / f"{image.frame_number:06d}.png")


rgb_camera.listen(save_rgb_image)
semseg_camera.listen(save_semseg_image)

# Simulate and record for 10 seconds at 20 FPS
simulation_duration = 10
fps = 20
frame_count = simulation_duration * fps

for _ in range(frame_count):
    world.tick()
    time.sleep(1 / 10)

print("destroy1")

# Destroy the actors (vehicles, pedestrians, and cameras)
#for vehicle in vehicles:
#    vehicle.destroy()
#
#print("destroy2")
#
#for pedestrian in pedestrians:
#    pedestrian.destroy()
#
#print("destroy3")
#
#rgb_camera.destroy()
#semseg_camera.destroy()

print(f"Number of RGB frames: {len(rgb_image_paths)}")
print(f"Number of SemSeg frames: {len(semseg_image_paths)}")



# Combine the RGB images into an MP4 video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rgb_video = cv2.VideoWriter('rgb_video.mp4', fourcc, fps, (512, 256))

for rgb_frame_path in rgb_image_paths:
    rgb_frame = cv2.imread(str(rgb_frame_path))
    print(f"Processing RGB frame: {rgb_frame_path}")
    if rgb_frame is not None:
        rgb_video.write(rgb_frame)
    else:
        print(f"Error reading RGB frame: {rgb_frame_path}")

# Release the RGB video writer
rgb_video.release()

# Combine the semantic segmentation images into an MP4 video
semseg_video = cv2.VideoWriter('semseg_video.mp4', fourcc, fps, (512, 256))

for semseg_frame_path in semseg_image_paths:
    semseg_frame = cv2.imread(str(semseg_frame_path))
    print(f"Processing semseg_frame_path frame: {semseg_frame_path}")

    if semseg_frame is not None:
        semseg_video.write(semseg_frame)
    else:
        print(f"Error reading semantic segmentation frame: {semseg_frame_path}")

# Release the semantic segmentation video writer
semseg_video.release()