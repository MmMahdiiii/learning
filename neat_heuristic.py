import neat
import os
import pickle
import subprocess
import random
import sys
import json
import os
import threading
import time
import json

directory = os.getcwd()
server_directory = directory + '/PythonServer'
client_directory = directory + '/PythonClient'
num_servers = 500

# TODO : run num_servers servers (pop_size % num_servers == 0)
# TODO : each neural network would dump to a file
# TODO : generate python client for each neural network
# TODO : run num_servers clients (pop_size % num_servers == 0)
# TODO : each client would load the neural network from the file
# TODO : we have to assign each client to a server (port)
# TODO : they have to play against two zigzag bot and GeneticMinimax bot

maps = []


def init_maps():
    maps_path = server_directory + '/maps/'
    for filename in os.listdir(maps_path):
        maps.append('maps/' + filename)


def generate_servers_config(chosen_map):
    #     read the config gamecfg.json
    #     change the ports of the configs to be different
    server_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gamecfg.json"
    )
    with open(server_config_path, 'r') as f:
        config = json.load(f)
    for i in range(num_servers):
        config['net']['port'] = 8000 + i
        #        write the config to a new file
        file_name = 'gamecfg' + str(i) + '.json'
        # dump the configs to server_configs folder
        config['game_handler']['map'] = chosen_map

        file_path = server_directory + '/server_configs/' + file_name
        with open(file_path, 'w') as outfile:
            json.dump(config, outfile)


def run_server(_map):
    #   give each server a config file_path as an argument
    #   change the directory to the server directory
    generate_servers_config(_map)
    os.chdir(server_directory)
    server_processes = []
    for i in range(num_servers):
        file_name = 'gamecfg' + str(i) + '.json'
        file_path = 'server_configs/' + file_name
        server_processes.append(subprocess.Popen(['python', 'main.py', file_path]))
    return server_processes


def run_games(genomes, config):
    #     dump each nn to a file with name nn_i in nn folder
    #     generate a config file for each client in configs folder
    #     relocate to the client directory
    #     run the clients

    os.chdir(client_directory)

    with open('gamecfg2.json', 'r') as sample_config:
        client_config = json.load(sample_config)

    results = [open('client_results/client' + str(i) + '.txt', 'w') for i in range(len(genomes))]
    gens = []
    processes = []
    counter = 0
    for genome_id, genome in genomes:
        gens.append(genome)
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        file_name = 'nn_' + str(counter) + '.pkl'
        file_path = 'nn/' + file_name
        with open(file_path, 'wb') as output:
            pickle.dump(net, output, 1)

        # generate a config file for each client in configs folder
        client_config_path = 'configs/' + 'gamecfg' + str(counter) + '.json'
        client_config['ai']['nn_path'] = file_path
        client_config['net']['port'] = 8000 + counter % num_servers
        client_config['ai']['stdout'] = 'client_results/client' + str(counter) + '.txt'

        with open(client_config_path, 'w') as outfile:
            json.dump(client_config, outfile)

        p = subprocess.Popen(['python', 'main.py', client_config_path], stdout=results[counter])
        processes.append(p)
        counter += 1

    scores = [0 for i in range(len(genomes))]
    time.sleep(5)
    for i, p in enumerate(processes):
        # if taking too long, kill the process
        try:
            p.wait(800)
        except subprocess.TimeoutExpired:
            pass
        results[i].close()

    for i, p in enumerate(processes):
        try:
            with open('client_results/client' + str(i) + '.txt', 'r') as f:
                lines = f.readlines()
                #           Side: Yellow
                #           Side: Blue

                color = 'Yellow'
                for line in lines:
                    if line.startswith('Side: '):
                        color = line.split()[1]
                    if line.startswith('    Blue -> '):
                        blue_score = int(line.split('-> ')[1])
                        if color == 'Blue':
                            scores[i] += blue_score
                    if line.startswith('    Yellow -> '):
                        yellow_score = int(line.split('-> ')[1])
                        if color == 'Yellow':
                            scores[i] += yellow_score
        except:
            pass
        

    for i, genome in enumerate(gens):
        print('genome ', i, ' score: ', scores[i])
        genome.fitness = scores[i]


best_scores = []

gen = 1


def eval_genomes(genomes, config):
    global gen, best_scores

    # select a random map
    chosen_map = random.choice(maps)
    # run the servers
    genome_groups = []
    server_processes = []
    for slide in range(0, len(genomes), 2 * num_servers):
        genome_groups.append(genomes[slide: slide + 2 * num_servers])
    for genome_group in genome_groups:
        server_processes = run_server(chosen_map)
        time.sleep(10)
        run_games(genome_group, config)

    # saving the best genome for each 2 generations
    # saving the best fitness for each generation
    best_fitness = 0
    best_genome = None
    for genome_id, genome in genomes:
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome
        best_scores.append(best_fitness)

    os.chdir(directory)
    file_name = 'best_genome_' + str(gen) + '.pkl'
    file_path = 'solutions/' + file_name
    with open(file_path, 'wb') as output:
        pickle.dump(best_genome, output, 1)
    nn = neat.nn.FeedForwardNetwork.create(best_genome, config)
    file_name = 'best_nn_' + str(gen) + '.pkl'
    file_path = 'solutions/' + file_name
    with open(file_path, 'wb') as output:
        pickle.dump(nn, output, 1)
    # storing scores
    with open('best_scores.txt', 'w') as f:
        for score in best_scores:
            f.write(str(score) + '\n')

    # killing the servers
    for p in server_processes:
        p.kill()

    gen += 1

    time.sleep(5)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    init_maps()
    run(config_path)
    print(best_scores)
