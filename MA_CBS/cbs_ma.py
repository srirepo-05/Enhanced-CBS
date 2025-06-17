import time as timer
import heapq
import random
from multi_agent_planner import ma_star, get_sum_of_cost, compute_heuristics, get_location
import copy
import numpy


def detect_collision(path1, path2, pos=None):
    """Return the first collision that occurs between two robot paths"""
    if pos is None:
        pos = []
    t_range = max(len(path1), len(path2))
    for t in range(t_range):
        loc_c1 = get_location(path1, t)
        loc_c2 = get_location(path2, t)
        loc1 = get_location(path1, t+1)
        loc2 = get_location(path2, t+1)
        
        # vertex collision
        if loc1 == loc2:
            pos.append(loc1)
            return pos, t
        # edge collision
        if [loc_c1, loc1] == [loc2, loc_c2]:
            pos.append(loc2)
            pos.append(loc_c2)
            return pos, t
    return None


def detect_collisions(paths, ma_list, collisions=None):
    """Return a list of first collisions between all robot pairs"""
    if collisions is None:
        collisions = []
    for ai in range(len(paths)-1):
        for aj in range(ai+1, len(paths)):
            collision_result = detect_collision(paths[ai], paths[aj])
            if collision_result is not None:
                position, t = collision_result

                ma_i = get_ma_of_agent(ai, ma_list)
                ma_j = get_ma_of_agent(aj, ma_list)

                # check if internal collision in the same meta-agent
                if ma_i != ma_j:
                    collisions.append({'a1': ai, 'ma1': ma_i,
                                     'a2': aj, 'ma2': ma_j,
                                     'loc': position,
                                     'timestep': t+1})
    return collisions


def count_all_collisions_pair(path1, path2):
    """Count all collisions between two paths"""
    collisions = 0
    t_range = max(len(path1), len(path2))
    for t in range(t_range):
        loc_c1 = get_location(path1, t)
        loc_c2 = get_location(path2, t)
        loc1 = get_location(path1, t+1)
        loc2 = get_location(path2, t+1)
        if loc1 == loc2 or [loc_c1, loc1] == [loc2, loc_c2]:
            collisions += 1
    return collisions


def count_all_collisions(paths):
    """Count all collisions in paths"""
    collisions = 0
    for i in range(len(paths)-1):
        for j in range(i+1, len(paths)):
            ij_collisions = count_all_collisions_pair(paths[i], paths[j])
            collisions += ij_collisions
    return collisions    


def standard_splitting(collision, constraints=None):
    """Return constraints to resolve collision using standard splitting"""
    if constraints is None:
        constraints = []

    if len(collision['loc']) == 1:  # vertex collision
        constraints.append({'agent': collision['a1'],
                           'meta_agent': collision['ma1'],
                           'loc': collision['loc'],
                           'timestep': collision['timestep'],
                           'positive': False})
        constraints.append({'agent': collision['a2'],
                           'meta_agent': collision['ma2'],
                           'loc': collision['loc'],
                           'timestep': collision['timestep'],
                           'positive': False})
    else:  # edge collision
        constraints.append({'agent': collision['a1'],
                           'meta_agent': collision['ma1'],
                           'loc': [collision['loc'][0], collision['loc'][1]],
                           'timestep': collision['timestep'],
                           'positive': False})
        constraints.append({'agent': collision['a2'],
                           'meta_agent': collision['ma2'],
                           'loc': [collision['loc'][1], collision['loc'][0]],
                           'timestep': collision['timestep'],
                           'positive': False})
    return constraints


def disjoint_splitting(collision, constraints=None):
    """Return constraints to resolve collision using disjoint splitting"""
    if constraints is None:
        constraints = []

    a = random.choice([('a1', 'ma1'), ('a2', 'ma2')])
    agent = a[0]
    meta_agent = a[1]

    if len(collision['loc']) == 1:  # vertex collision
        constraints.append({'agent': collision[agent],
                           'meta_agent': collision[meta_agent],
                           'loc': collision['loc'],
                           'timestep': collision['timestep'],
                           'positive': True})
        constraints.append({'agent': collision[agent],
                           'meta_agent': collision[meta_agent],
                           'loc': collision['loc'],
                           'timestep': collision['timestep'],
                           'positive': False})
    else:  # edge collision
        if agent == 'a1':
            constraints.append({'agent': collision[agent],
                               'meta_agent': collision[meta_agent],
                               'loc': [collision['loc'][0], collision['loc'][1]],
                               'timestep': collision['timestep'],
                               'positive': True})
            constraints.append({'agent': collision[agent],
                               'meta_agent': collision[meta_agent],
                               'loc': [collision['loc'][0], collision['loc'][1]],
                               'timestep': collision['timestep'],
                               'positive': False})
        else:
            constraints.append({'agent': collision[agent],
                               'meta_agent': collision[meta_agent],
                               'loc': [collision['loc'][1], collision['loc'][0]],
                               'timestep': collision['timestep'],
                               'positive': True})
            constraints.append({'agent': collision[agent],
                               'meta_agent': collision[meta_agent],
                               'loc': [collision['loc'][1], collision['loc'][0]],
                               'timestep': collision['timestep'],
                               'positive': False})
    return constraints


def get_ma_of_agent(agent, ma_list):
    """Get the meta-agent an agent is part of"""
    for ma in ma_list:
        if agent in ma:
            return ma
    raise BaseException('No meta-agent found for agent')


def meta_agents_violate_constraint(constraint, paths, ma_list, violating_ma=None):
    """Find meta-agents that violate a positive constraint"""
    assert constraint['positive'] is True
    if violating_ma is None:
        violating_ma = []

    for i in range(len(paths)):
        ma_i = get_ma_of_agent(i, ma_list)

        if ma_i == constraint['meta_agent'] or ma_i in violating_ma:
            continue

        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                violating_ma.append(ma_i)
        else:  # edge constraint
            if (constraint['loc'][0] == prev or constraint['loc'][1] == curr 
                or constraint['loc'] == [curr, prev]):
                violating_ma.append(ma_i)

    return violating_ma


def combined_constraints(constraints, new_constraints, updated_constraints=None):
    """Combine existing constraints with new ones"""
    if isinstance(new_constraints, list):
        updated_constraints = copy.deepcopy(new_constraints)
    else:
        updated_constraints = [new_constraints]

    for c in constraints:
        if c not in updated_constraints:
            updated_constraints.append(c)

    return updated_constraints


def bypass_found(curr_cost, new_cost, curr_collisions_num, new_collisions_num):
    """Check if bypass is found"""
    if curr_cost == new_cost and (new_collisions_num < curr_collisions_num):
        return True
    return False


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """Initialize CBS solver
        my_map   - list of lists specifying obstacle positions
        starts   - [(x1, y1), (x2, y2), ...] list of start locations
        goals    - [(x1, y1), (x2, y2), ...] list of goal locations
        """
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        """Push node to open list"""
        heapq.heappush(self.open_list, (node['cost'], len(node['ma_collisions']), self.num_of_generated, node))
        print("> Generate node {} with cost {}".format(self.num_of_generated, node['cost']))
        self.num_of_generated += 1

    def pop_node(self):
        """Pop node from open list"""
        _, _, id, node = heapq.heappop(self.open_list)
        print("> Expand node {} with cost {}".format(id, node['cost']))
        self.num_of_expanded += 1
        return node

    def empty_tree(self):
        """Empty the search tree"""
        self.open_list.clear()

    def detect_cardinal_conflict(self, p, collision):
        """Detect if conflict is cardinal, semi-cardinal, or non-cardinal"""
        cardinality = 'non-cardinal'

        # temporary constraints for detecting cardinal collision
        temp_constraints = standard_splitting(collision)

        ma1 = collision['ma1']
        
        # Test first meta-agent
        path1_constraints = combined_constraints(p['constraints'], temp_constraints[0])
        alt_paths1 = ma_star(self.my_map, self.starts, self.goals, self.heuristics,
                            list(ma1), path1_constraints)

        # get current paths of meta-agent
        curr_paths = []
        for a1 in ma1:
            curr_paths.append(p['paths'][a1])

        curr_cost = get_sum_of_cost(curr_paths)
        alt_cost = 0
        if alt_paths1:
            alt_cost = get_sum_of_cost(alt_paths1)

        if not alt_paths1 or alt_cost > curr_cost:
            cardinality = 'semi-cardinal'

        ma2 = collision['ma2']
        
        # Test second meta-agent
        path2_constraints = combined_constraints(p['constraints'], temp_constraints[1])
        alt_paths2 = ma_star(self.my_map, self.starts, self.goals, self.heuristics,
                            list(ma2), path2_constraints)
        
        curr_paths = []
        for a2 in ma2:
            curr_paths.append(p['paths'][a2])

        curr_cost = get_sum_of_cost(curr_paths)
        alt_cost = 0
        if alt_paths2:
            alt_cost = get_sum_of_cost(alt_paths2)

        if not alt_paths2 or alt_cost > curr_cost:
            if cardinality == 'semi-cardinal':
                cardinality = 'cardinal'
            else:
                cardinality = 'semi-cardinal'

        return cardinality

    def should_merge(self, collision, p, N=0):
        """Check if meta-agents should be merged"""
        CM = 0
        ma1 = collision['ma1']
        ma2 = collision['ma2']
        
        for ai in ma1:
            for aj in ma2:
                if ai > aj:
                    ai, aj = aj, ai
                CM += p['agent_collisions'][ai][aj]
        
        if CM > N:
            print('> Merge meta-agents {}, {} into one meta-agent'.format(ma1, ma2))
            return True
        return False

    def generate_child(self, constraints, paths, agent_collisions, ma_list):
        """Generate child node"""
        collisions = detect_collisions(paths, ma_list)
        cost = get_sum_of_cost(paths)
        child_node = {
            'cost': cost,
            'constraints': copy.deepcopy(constraints),
            'paths': copy.deepcopy(paths),
            'ma_collisions': collisions,
            'agent_collisions': copy.deepcopy(agent_collisions),
            'ma_list': copy.deepcopy(ma_list)
        }
        return child_node

    def merge_agents(self, collision, ma_list):
        """Merge agents into a single meta-agent"""
        ma1 = collision['ma1']
        ma2 = collision['ma2']
        meta_agent = set.union(ma1, ma2)

        print('new merged meta_agent ', meta_agent)

        ma_list.remove(ma1)
        ma_list.remove(ma2)
        ma_list.append(meta_agent)

        return meta_agent, ma_list

    def find_solution(self, disjoint=False):
        """Find paths for all agents from start to goal locations"""
        self.start_time = timer.time()
        
        if disjoint:
            splitter = disjoint_splitting
        else:
            splitter = standard_splitting

        print("USING: ", splitter.__name__)

        # Generate root node
        root = {
            'cost': 0,
            'constraints': [],
            'paths': [],
            'ma_collisions': [],
            'agent_collisions': numpy.zeros((self.num_of_agents, self.num_of_agents)),
            'ma_list': []
        }

        # Find initial path for each agent
        for i in range(self.num_of_agents):
            path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,
                          [i], root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['ma_list'].append({i})
            root['paths'].extend(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['ma_collisions'] = detect_collisions(root['paths'], root['ma_list'])
        self.push_node(root)

        # Main CBS loop
        while len(self.open_list) > 0:
            if self.num_of_generated > 50000:
                print('reached maximum number of nodes. Returning...')
                return None

            p = self.pop_node()
            
            if p['ma_collisions'] == []:
                self.print_results(p)
                return p['paths'], self.num_of_generated, self.num_of_expanded

            print('Node expanded. Collisions: ', len(p['ma_collisions']))

            # Select collision to resolve
            chosen_collision = None
            collision_type = 'non-cardinal'
            
            # Look for cardinal conflicts first
            for collision in p['ma_collisions']:
                collision_type = self.detect_cardinal_conflict(p, collision)
                if collision_type == 'cardinal':
                    chosen_collision = collision
                    break

            # If no cardinal, look for semi-cardinal
            if chosen_collision is None:
                for collision in p['ma_collisions']:
                    collision_type = self.detect_cardinal_conflict(p, collision)
                    if collision_type == 'semi-cardinal':
                        chosen_collision = collision
                        break

            # If none found, pick first collision
            if chosen_collision is None:
                chosen_collision = p['ma_collisions'][0]

            # Update collision history
            chosen_a1 = chosen_collision['a1']
            chosen_a2 = chosen_collision['a2']
            if chosen_a1 > chosen_a2:
                chosen_a1, chosen_a2 = chosen_a2, chosen_a1
            p['agent_collisions'][chosen_a1][chosen_a2] += 1

            # Generate constraints
            new_constraints = splitter(chosen_collision)
            
            child_nodes = []
            bypass_successful = False

            for constraint in new_constraints:
                updated_constraints = combined_constraints(p['constraints'], constraint)
                q = self.generate_child(updated_constraints, p['paths'], 
                                       p['agent_collisions'], p['ma_list'])

                ma = constraint['meta_agent']

                # Find new path for constrained meta-agent
                path = ma_star(self.my_map, self.starts, self.goals, self.heuristics,
                              list(ma), q['constraints'])

                if constraint['positive']:
                    assert path

                if path is not None:
                    # Update paths
                    for i, agent in enumerate(ma):
                        q['paths'][agent] = path[i]

                    # Handle positive constraints
                    if constraint['positive']:
                        violating_ma_list = meta_agents_violate_constraint(constraint, q['paths'], q['ma_list'])
                        no_solution = False
                        
                        for v_ma in violating_ma_list:
                            v_ma_list = list(v_ma)
                            path_v_ma = ma_star(self.my_map, self.starts, self.goals, 
                                              self.heuristics, v_ma_list, q['constraints'])
                            
                            if path_v_ma is not None:
                                for i, agent in enumerate(v_ma_list):
                                    q['paths'][agent] = path_v_ma[i]
                            else:
                                no_solution = True
                                break

                        if no_solution:
                            continue

                    q['ma_collisions'] = detect_collisions(q['paths'], q['ma_list'])
                    q['cost'] = get_sum_of_cost(q['paths'])

                    # Check for bypass
                    if (collision_type != 'cardinal' and 
                        bypass_found(p['cost'], q['cost'], len(p['ma_collisions']), len(q['ma_collisions']))):
                        print('> Take Bypass')
                        self.push_node(q)
                        bypass_successful = True
                        break

                    child_nodes.append(copy.deepcopy(q))

            if bypass_successful:
                continue

            # MA-CBS merging
            if self.should_merge(chosen_collision, p):
                meta_agent, updated_ma_list = self.merge_agents(chosen_collision, p['ma_list'])

                # Update paths
                meta_agent_paths = ma_star(self.my_map, self.starts, self.goals, self.heuristics,
                                         list(meta_agent), p['constraints'])

                if meta_agent_paths:
                    updated_paths = copy.deepcopy(p['paths'])

                    for i, agent in enumerate(meta_agent):
                        updated_paths[agent] = meta_agent_paths[i]

                    # Update collisions, cost
                    updated_node = self.generate_child(p['constraints'], updated_paths, 
                                                     p['agent_collisions'], updated_ma_list)

                    # Merge & restart
                    self.empty_tree()
                    self.push_node(updated_node)
                    continue

            # Push child nodes
            for n in child_nodes:
                self.push_node(n)

        return None

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])