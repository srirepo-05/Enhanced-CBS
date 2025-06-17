#!/usr/bin/python
import argparse
from eecbs import EECBSSolver
from eecbs_dc import EECBS_DC
from mleecbs import MLEECBSSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def __getitem__(self, index):
        return [self.x, self.y][index]
    
    def __repr__(self):
        return f"({self.x}, {self.y})"

class Agent:
    def __init__(self, id, start_pos, goal_pos):
        self.id = id
        self.start = start_pos
        self.goal = goal_pos
    
    def __repr__(self):
        return f"Agent {self.id}: {self.start} -> {self.goal}"

def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)

def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)

def main():
    # Define the custom maze (True = obstacle, False = free space)
    # This is a large 32x32 maze - format: True for obstacles, False for free space
    # You can replace this with your own maze definition
    my_map = [
        # [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        # ... (32 rows of 32 boolean values each)
        # Format: True = obstacle (@), False = free space (.)
        # For now, using a simple 8x8 map for testing
        
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    [True, False, False, True, False, False, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, False, False, True, False],
    [True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, True, False, False, True, False],
    [True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, True, False, False, True, False],
    [True, False, False, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, False, False, True, False, False, True, True, True, True, False],
    [True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False],
    [True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, False, False, True, True, True, True, False, False, True, True, True, True, True],
    [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, True, False, False, False, False],
    [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, True, False, False, False, False],
    [True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, False, False, True, True, True, True, False],
    [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False],
    [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False],
    [True, False, False, True, True, True, True, True, True, True, False, False, True, True, True, True, False, False, True, False, False, True, True, True, True, False, False, True, False, False, True, True],
    [True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False],
    [True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False],
    [True, True, True, True, False, False, True, True, True, True, False, False, True, False, False, True, True, True, True, False, False, True, True, True, True, True, True, True, False, False, True, False],
    [True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False],
    [True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False],
    [True, False, False, True, False, False, True, True, True, True, True, True, True, False, False, True, False, False, True, True, True, True, False, False, True, True, True, True, True, True, True, False],
    [True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False, False, False, False, False],
    [True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False, False, False, False, False],
    [True, False, False, True, True, True, True, False, False, True, True, True, True, False, False, True, True, True, True, False, False, True, False, False, True, True, True, True, True, True, True, False],
    [True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False],
    [True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False],
    [True, False, False, True, False, False, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, False, False, True, True, True, True, False],
    [True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False],
    [True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, False],
    [True, False, False, True, False, False, True, True, True, True, False, False, True, False, False, True, False, False, True, True, True, True, False, False, True, True, True, True, False, False, True, False],
    [True, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False]


    ]
    
    # Define agents with start and goal positions
    # Format: Agent(id, Position(row, col), Position(row, col))
    agents = [
            Agent(1, Position(17, 21), Position(15, 16)),
            Agent(2, Position(23, 23), Position(10, 19)),
            Agent(3, Position(1, 20), Position(8, 2)),
            Agent(4, Position(2, 29), Position(31, 22)),
            Agent(5, Position(28, 26), Position(25, 16)),
            Agent(6, Position(13, 5), Position(5, 1)),
            Agent(7, Position(13, 9), Position(14, 19)),
            Agent(8, Position(1, 29), Position(14, 8)),
            Agent(9, Position(14, 24), Position(19, 4)),
            Agent(10, Position(17, 19), Position(25, 14)),
        # Add more agents as needed
        # Agent(4, Position(row, col), Position(row, col)),
    ]
    
    print("Map:")
    for i, row in enumerate(my_map):
        print(f"Row {i}: {row}")
    
    print("\nAgents:")
    for agent in agents:
        print(agent)
    
    # Extract start and goal positions
    starts = [(agent.start.x, agent.start.y) for agent in agents]  
    goals = [(agent.goal.x, agent.goal.y) for agent in agents]    
    
    print(f"\nStarts: {starts}")
    print(f"Goals: {goals}")
    
    # Print the MAPF instance
    print("\n***MAPF Instance***")
    print_mapf_instance(my_map, starts, goals)
    
    # Parse command line arguments for solver selection
    parser = argparse.ArgumentParser(description='Runs EECBS algorithm variants')
    parser.add_argument('--solver', type=str, default='EECBS',
                        help='The solver to use (one of: {EECBS, EECBSDC, MLEECBS}), defaults to EECBS')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Save the solved path animation as a gif')
    parser.add_argument('--map_name', type=str, default='custom_map',
                        help='Name for the map (used by some solvers)')
    
    args = parser.parse_args()
    
    # Select and run the appropriate solver
    if args.solver == "EECBS":
        print("\n***Run EECBS***")
        solver = EECBSSolver(my_map, starts, goals)
        paths = solver.find_solution()
    elif args.solver == "EECBSDC":
        print("\n***Run EECBS Data Collection***")
        solver = EECBS_DC(my_map, starts, goals, args.map_name)    
        paths = solver.find_solution()
    elif args.solver == "MLEECBS":
        print("\n***Run MLEECBS***")
        solver = MLEECBSSolver(my_map, starts, goals, args.map_name, args.batch)
        paths = solver.find_solution()
    else:
        raise RuntimeError("Unknown solver! Use one of: EECBS, EECBSDC, MLEECBS")
    
    if paths:
        # Calculate and print the total cost
        cost = get_sum_of_cost(paths)
        print(f"\nSolution found with total cost: {cost}")
        
        # Print the final paths
        print(f"\nFinal paths found:")
        for i, path in enumerate(paths):
            print(f"Agent {i}: {path}")
        
        # Show visualization if not in batch mode
        if not args.batch:
            print("\n***Test paths on a simulation***")
            animation = Animation(my_map, starts, goals, paths)
            if args.save:
                animation.save("eecbs_animation.gif", speed=1)
                print("Animation saved as eecbs_animation.gif")
            animation.show()
    else:
        print("No solution found!")

if __name__ == '__main__':
    main()