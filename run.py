
import sys
import neat
import pickle
from Game import *

def NEAT_Training(genomes, config):
	"""
	Executes the NEAT training algoithmn
	"""
		
	# Initalise Game
	game = Game(train=True, Human=False)

	# Initalise Genome Variables
	nets, cars = [], []
	for id, g in genomes:
		cars.append(game.create_AI())
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		g.fitness = 0
	game.set_focus_car(cars[0])

	# Main Game Loop
	count = 0
	game.AIs[0].update()
	while game.running:
		game.clock.tick(30)	
		game.process_events()
		game.camera_offset = game.camera.update(game.AIs[0])
		for index, car in enumerate(cars):

			# Process action for the car
			if car.is_alive():
				inputs = car.get_LIDAR()
				action = nets[index].activate(inputs)
				steering = - action[0] + action[1]
				speed_input = action[2]
				car.set_input(steering, speed_input)
			
				# Check if car still on the road
				for key, line in game.walls.items():
					if key == "OuterWall": # If inside the outerwall
						insideOuterWall = line.inside_polygon(car.rect.center)
					elif key == "InnerWall": # If outside the innerwall
						insideInnerWall = line.inside_polygon(car.rect.center)
				offRoad = insideOuterWall == insideInnerWall	
				if offRoad and count>1: car.kill()
				if car.laps_done() == 2: car.kill()

		# Update cars and assess fitness
		remain_cars = 0
		max_fitness = 0
		for i, car in enumerate(cars):
			if car.is_alive():
				remain_cars += 1
				car.update()
				genomes[i][1].fitness += car.get_reward()
				fitness = genomes[i][1].fitness
				if fitness > max_fitness*1.2:
					game.focus_car = car
					max_fitness = fitness
		if remain_cars == 0: break

		game.update()
		game.draw()
		count += 1
	pg.quit()
	return min([g.fitness for id, g in genomes])

def NEAT_Run(config):
	"""
	Runs the best stored NEAT implementation
	"""

	# Initalise Game
	game = Game(lidar=True, Human=False)

	# Load the Winner
	with open('winner', 'rb') as f:
		c = pickle.load(f)
	print('Loaded genome:\n',c)

	net = neat.nn.FeedForwardNetwork.create(c, config)

	count = 0
	car = game.create_AI()
	game.set_focus_car(car)
	car.update()
	while game.running:

		# Update game clock & camera positionG
		game.clock.tick(60)
		game.process_events()
		game.camera_offset = game.camera.update(car)

		# Calculate & apply next action
		inputs = car.get_LIDAR()
		action = net.activate(inputs)
		steering = - action[0] + action[1]
		speed_input = action[2]
		car.set_input(steering, speed_input)
		car.update()
		if car.laps_done() == 2: break

		# Update Game State
		game.update()
		game.draw()
		count += 1
		
	text_flag = True
	while game.running:
		game.process_events()
		if text_flag:
			game.text.draw_endscreen(game.screen, 1)
			pg.display.flip()
			text_flag = False
	pg.quit()


if __name__ == '__main__':
	
	arg = sys.argv[1]
	if 'human' in arg:
			game = Game(AI=False)
			car = game.create_Human()
			game.set_focus_car(car)
			game.run()
	else:
		config_path = "./config"
		config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
						neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
		if 'train' in arg:
			# Load Configuration Files
			p = neat.Population(config)
			p.add_reporter(neat.StdOutReporter(True))
			stats = neat.StatisticsReporter()
			
			# Train Cars with NEAT
			p.add_reporter(stats)
			winner = p.run(NEAT_Training, 1000)

			# Save the Winner
			with open('winner-test', 'wb') as f:
				pickle.dump(winner, f)
			print(winner)
		elif 'robot' in arg:
			NEAT_Run(config)
		else:
			print("Please enter 'human, train or robot' as a valid arg.")

