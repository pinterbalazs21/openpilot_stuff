import carla
import pygame
import numpy as np

def main():
    # Connect to the CARLA server
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Get the world and map
    world = client.get_world()
    carla_map = world.get_map()

    # Set up the Pygame window for visualization
    pygame.init()
    width, height = 800, 600
    display = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    while True:
        # Check for Pygame events (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Get the map image from the CARLA API
        map_image = carla_map.to_image(carla_map.width, carla_map.height, True)

        # Convert the map image to a format suitable for Pygame
        map_image_np = np.frombuffer(map_image.raw_data, dtype=np.uint8)
        map_image_np = map_image_np.reshape((map_image.height, map_image.width, 4))
        map_image_np = np.flip(map_image_np, axis=0)
        map_image_surface = pygame.surfarray.make_surface(map_image_np)

        # Display the map in the Pygame window
        display.blit(map_image_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
