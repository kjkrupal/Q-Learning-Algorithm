import numpy as np

def create_environment(grid_size, wall_coordinates, goal_coordinates, nega_coordinates):
    
    environment = np.zeros(grid_size)
    environment = np.pad(environment, 1, 'constant', constant_values = None)

    for coordinate in wall_coordinates:
        environment[coordinate] = None
    
    for coordinate in goal_coordinates:
        environment[coordinate] = 1
    
    for coordinate in nega_coordinates:
        environment[coordinate] = -1
    
    print(environment)
    np.save('Environments/grid_game', environment)
    np.save('Environments/wall_coordinates', np.array(wall_coordinates))
    np.save('Environments/goal_coordinates', np.array(goal_coordinates))
    np.save('Environments/nega_coordinates', np.array(nega_coordinates))

def update_coordinates(coordinates):
    new_list = []
    for a in coordinates:
        a = list(a)
        a[0] = a[0] + 1
        a[1] = a[1] + 1
        a = tuple(a)
        new_list.append(a)
    return new_list

def main():
    nega_coordinates =  update_coordinates([(3,3), (4,5), (4,6), (5,6), (5,8), (6,8), (7,3), (7,5), (7,6)])
    wall_coordinates =  update_coordinates([(2,1), (2,2), (2,3), (2,4), (2,6), (2,7), (2,8), (3,4), (4,4), (5,4), (6,4), (7,4)])
    goal_coordinates =  update_coordinates([(5,5)])
    grid_size = (10, 10)
    create_environment(grid_size, wall_coordinates, goal_coordinates, nega_coordinates)

if __name__ == "__main__":
    main()