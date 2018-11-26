import numpy as np

def create_environment(grid_size, wall_coordinates, goal_coordinates, nega_coordinates):
    
    environment = np.zeros(grid_size)
    
    for coordinate in wall_coordinates:
        environment[coordinate] = None
    
    for coordinate in goal_coordinates:
        environment[coordinate] = 1
    
    for coordinate in nega_coordinates:
        environment[coordinate] = -1
        
    np.save('Environments/grid_game', environment)

def main():
    nega_coordinates = [(3,3), (4,5), (4,6), (5,6), (5,8), (6,8), (7,3), (7,5), (7,6)]
    wall_coordinates = [(2,1), (2,2), (2,3), (2,4), (2,6), (2,7), (2,8), (3,4), (4,4), (5,4), (6,4), (7,4)]
    goal_coordinates = [(5,5)]
    grid_size = (10, 10)
    create_environment(grid_size, wall_coordinates, goal_coordinates, nega_coordinates)

if __name__ == "__main__":
    main()