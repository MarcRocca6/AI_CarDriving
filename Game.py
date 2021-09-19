#!/usr/bin/env python3

"""
Creates a car driving game that can be interacted
with by a user or used to train a neural network.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pytmx 
import numpy as np
import pygame as pg
from random import randint
import matplotlib.path as mplPath
from math import pi, sin, cos, inf, log, sqrt

NUM_LAPS = 2
W_WIDTH = 800
W_HEIGHT = 800
CAR_WIDTH = 32
MAP_WIDTH = 2500
MAP_HEIGHT = 2500
NUM_CHECKPOINTS = 4
COLORS = ['blue','green','yellow','black']

class Game():
	"""
	Class containing the game and the current game state
	"""
		
	def __init__(self, train=False, lidar=False, AI=True, Human=True):
		"""
		Game Initalisation
		@param bool train: If the Game is being used for training the model
		@param bool lidar: Whether the game should display the lidar readings
		@param bool AI: Whether AI cars are to be included
		@param bool Human: Whether Human cars are to be included
		"""

		# Initalise PyGame Window
		pg.init()
		pg.display.set_caption("Moving Car")
		self.clock = pg.time.Clock()
		self.running = True  	
		self.train = train
		self.lidar = lidar

		# Create & Display Backgound
		self.screen = pg.display.set_mode((W_WIDTH, W_HEIGHT))
		self.screen_rect = self.screen.get_rect()
		self.camera = Camera(MAP_WIDTH, MAP_WIDTH)
		self.camera_offset = (0,0)
		pg.display.flip()

		# Create Map
		if train: 
			mapPath = 'assets/Map_Train.tmx'
			global NUM_CHECKPOINTS
			NUM_CHECKPOINTS = 15
		else: mapPath = 'assets/Map_Run.tmx'
		self.map = TiledMap(mapPath)
		self.map_img = self.map.make_map()
		self.map_rect = self.map_img.get_rect()
		
		# Load Map Objects
		self.blocks, self.checkpoints, spawn_points, self.walls = {},{},{},{}
		msf = self.map.map_scale_ratio
		for tile_object in self.map.tmxdata.objects:
			if ('spawn' in tile_object.name.lower()):
				spawn_points[tile_object.name] = pg.math.Vector2(int(tile_object.x*msf), int(tile_object.y*msf))
			else:
				points = [pg.math.Vector2(p)*msf for p in tile_object.points]	
				if ('wall' in tile_object.name.lower()):
					self.walls[tile_object.name] = Line(points) 	
				elif ('block' in tile_object.name.lower()):
					self.blocks[tile_object.name] = Line(points) 
				elif ('checkpoint' in tile_object.name.lower()):
					self.checkpoints[int(tile_object.name[-2:])] = Line(points) 

		# Add all sprites
		self.race_won = None
		self.AIs, self.Humans = [], []
		self.all_sprites = pg.sprite.RenderPlain()
		self.AI_spawn, self.Human_spawn = None, None
		for key, pt in spawn_points.items():
			if 'AI' in key and AI: self.AI_spawn = pt
			elif 'Human' in key and Human: self.Human_spawn = pt
		self.checkpoint_flash = Checkpoints(self.checkpoints)
		self.text = Text()

	def create_AI(self):
		"""
		Initalise and create an AI instance of a car
		"""
		car = Car_AI(self.AI_spawn, -90, self.blocks, self.checkpoints, self.walls, color=randint(1,3))
		self.AIs.append(car)
		self.all_sprites.add(car)
		return car

	def create_Human(self):
		"""
		Initalise and create a human instance of a car
		"""
		car = Car(self.Human_spawn, -90, self.blocks, self.checkpoints, color=randint(1,3))
		self.Humans.append(car)
		self.all_sprites.add(car)
		return car

	def run(self):
		"""
		Executes the main game loop.
		"""
		while self.running:
			self.clock.tick(60)	
			self.process_events()
			self.update()
			self.draw()
			if self.race_won != None: break
		text_flag = True
		while self.running:
			self.process_events()
			if text_flag:
				self.text.draw_endscreen(self.screen, self.race_won)
				pg.display.flip()
				text_flag = False
		pg.quit()
		
	def process_events(self):
		"""
		Process all global game events
		"""
		for event in pg.event.get():
			if event.type == pg.QUIT:
				self.running = False
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_ESCAPE:
					self.running = False	

	def update(self):
		"""
		Process game events for all sprites
		"""
		self.camera_offset = self.camera.update(self.focus_car)
		for i, sprite in enumerate(self.all_sprites.sprites()):
			if sprite.is_alive():
				offRoad = False
				if not sprite.is_AI(): 
					for key, line in self.walls.items():
						if key == "OuterWall": # If inside the outerwall
							insideOuterWall = line.inside_polygon(sprite.rect.center)
						elif key == "InnerWall": # If outside the innerwall
							insideInnerWall = line.inside_polygon(sprite.rect.center)
					offRoad = insideOuterWall == insideInnerWall
				sprite.update(offRoad = offRoad)
				if sprite.laps_done() == NUM_LAPS: self.race_won = i
			
	def draw(self):
		"""
		Draw the sprites & background into the game
		"""
		self.screen.blit(self.map_img, self.camera.apply_rect(self.map_rect))
		for sprite in self.all_sprites.sprites():
			if sprite.is_alive():
				self.screen.blit(sprite.image, self.camera.apply(sprite))
				if sprite.is_AI() and self.lidar:
			 		sprite.lidar.draw(self.screen, self.camera_offset)	
		self.text.draw(self.screen, sprite.laps_done())
		if not self.train:
			self.checkpoint_flash.draw(self.screen, self.camera_offset, 
				self.focus_car.checkpoints_passed%NUM_CHECKPOINTS+1)
		pg.display.flip()	

	def set_focus_car(self, sprite):
		"""
		Sets the car which the camera will focus upon 
		"""
		self.focus_car = sprite


class Text():
	"""
	Defines the functions to add text upon the screen
	"""
	
	def __init__(self):
		self.end_font_1 = pg.font.Font('assets/font1.otf',  40)
		self.end_font_2 = pg.font.Font('assets/paladin_c.otf',  50)
		self.font = pg.font.Font('assets/paladin_c.otf',  20)
		self.init_ticks = None
	
	def draw_endscreen(self, screen, car_num):
		"""
		Draws the text that will appear when a car has finished the game
		"""
		text_str_1 = f"Car   {car_num+1}   won"
		text_str_2 = f"{self.get_time()}"
		text_1 = self.end_font_1.render(text_str_1, False, pg.Color('black'))
		text_2 = self.end_font_2.render(text_str_2, False, pg.Color('black'))
		screen.blit(text_1,(W_WIDTH/4, W_HEIGHT/4))
		screen.blit(text_2,(W_WIDTH/3-30, W_HEIGHT/2))

	def draw(self, screen, lap_num):
		"""
		Draws the car status text onto the screen 
		"""
		text_str = f"Laps = {lap_num}/{NUM_LAPS}     Time = {self.get_time()}"
		text = self.font.render(text_str, False, pg.Color('white'))
		screen.blit(text,(10, 10))

	def get_time(self):
		"""
		Returns the time string representing how since the start of the game
		"""
		ticks = pg.time.get_ticks()
		if self.init_ticks == None: 
			self.init_ticks = ticks
		ticks -= self.init_ticks
		millis = ticks % 1000
		seconds = int(ticks/1000 % 60)
		minutes = int(ticks/60000 % 24)
		out ='{minutes:02d} : {seconds:02d} : {millis}'.format(minutes=minutes, 
			millis=millis, seconds=seconds)
		return out


class Checkpoints():
	"""
	Class defining where and how green checkpoints pictures will appear
	"""

	def __init__(self, checkpoints):
		self.image = pg.image.load('assets/Spritesheets/checkpoint.png').convert_alpha()
		self.image_copy = self.image
		self.locations = checkpoints
		self.last_shown_checkpoint = 0

	def draw(self, screen, camera_offset, checkpoint_number):
		"""
		Makes the respective green checkpoint appear on the map
		"""
		checkpoint = self.locations[int(checkpoint_number-1)]
		points = [p + pg.math.Vector2(camera_offset) for p in checkpoint.points]
		rect = update_rect(points)
		if checkpoint_number != self.last_shown_checkpoint:
			if rect.h > rect.w:
				self.image = pg.transform.rotate(self.image_copy, 90)
				w,h = rect.w, rect.h
			else: 
				self.image = self.image_copy
				w,h = rect.h, rect.w
			self.image = pg.transform.scale(self.image, (w,h))
			self.last_shown_checkpoint = checkpoint_number
		screen.blit(self.image, rect)


class Car(pg.sprite.Sprite):
	"""
	Wrapper for all sprite objects
	"""
	
	def __init__(self, point, degree, obstacles, checkpoints, color=1): 
		"""
		@param list points: pg.math.Vector2 points defining the shapes' polygon
		@param string imagePath: Filepath of the shapes' image
		@param int colour: Refers to the colour from the index of the global COLOR list
		"""
		# Car Rect and Edges Variables
		pg.sprite.Sprite.__init__(self)
		carPath = os.path.join(os.getcwd(), 'assets', 'Cars', 
			f"car_{COLORS[color]}_{randint(1,4)}.png")
		self.image = pg.image.load(carPath).convert_alpha()
		self.image = scale_image(self.image, CAR_WIDTH)
		self.image_copy = self.image.copy()
		self.rect = self.image.get_rect()
		self.obstacles = obstacles
		self.checkpoints = checkpoints
		self.checkpoints_passed = 0
		self.points = 	[self.rect.topleft, self.rect.topright, 
						self.rect.bottomright, self.rect.bottomleft]

		# Car Handling and Movement Variables 
		self.center = None				# Cur Center 
		self.heading = 0		 		# Cur Heading
		self.acc = pg.math.Vector2()	# Cur Acceleration
		self.vel = pg.math.Vector2()	# Cur Velocity

		self.AI = False
		self.reward = 0
		self.alive = True
		self.offRoad_friction = 1.55
		self.onRoad_friction = 0.9
		self.friction = 0.9 	 		# Cur Friction coeff
		self.d_accel = 15		 		# Max rate of car accleration
		self.d_rotation = 50	 		# Rate steering (low = more sensitive)
		self.d_t = (60.0/1000.0) 		# Change in time (FPS)
		self.accel_coeff = 0.3	 		# Acceleration Coeff
		self.max_vel = 8.571*(0.7/self.friction)*(self.d_accel/6)

		self.rotate(degree * (pi/180))
		self.move(point)

	def update(self, offRoad = False):
		"""
		Update the car's position from keyboard inputs
		"""
		if offRoad: self.friction = self.offRoad_friction
		else: 
			self.update_checkpoints()
			self.friction = self.onRoad_friction
		self.process_events()
		self.move(self.vel, offRoad = offRoad)	
		self.rect = update_rect(self.points)

	def process_inputs(self):
		"""
		Process the car's inputs
		"""
		# Get keyboard inputs
		keys = pg.key.get_pressed()
		rotation = keys[pg.K_RIGHT] - keys[pg.K_LEFT]
		linear = (keys[pg.K_DOWN]*0.40) - keys[pg.K_UP]
		#print(linear, rotation)
		return linear, rotation

	def process_events(self):
		"""
		Process keyboard events for the cars linear and rotational movement
		"""
		# Get keyboard inputs
		linear, rotation = self.process_inputs()

		# Execute Rotational Movement
		self.absolute_vel = self.vel.length()
		angle_deg = self.heading * 360 / (2*pi)
		angle_rad = (rotation * pi)*(self.absolute_vel / self.d_rotation)*self.d_t
		self.rotate(angle_rad)
		
		# Execute Linear Movement
		if linear > -0.1: self.acc.y = linear * self.d_accel # If going backwards
		else: self.acc.y = linear * self.get_gradual_accel() # If going forwards
		self.acc = self.acc.rotate(angle_deg) 
		self.vel += (self.acc - (self.friction*self.vel)) * self.d_t
		self.acc *= 0

	def move(self, displacement, offRoad = False):
		"""
		Move the shape by the point (x,y). Keep the car within the game boundaries
		"""
		points = []
		boundary_flag = False
		for p in self.points:
			newpoint = p + displacement
			if offRoad:
				# Makes sure vehicle is inside the map
				if newpoint.x < 0:
					displacement.x = 0-newpoint.x
					boundary_flag = True; break
				elif newpoint.x > MAP_WIDTH:
					displacement.x = MAP_WIDTH-newpoint.x
					boundary_flag = True; break
				elif newpoint.y < 0:
					displacement.y = 0-newpoint.y
					boundary_flag = True; break
				elif newpoint.y > MAP_HEIGHT:
					displacement.y = MAP_HEIGHT-newpoint.y
					boundary_flag = True; break

				if not self.is_AI():
					# Make sure vehicle will not collide with obstacles
					for key, obstacle in self.obstacles.items():
						if obstacle.inside_polygon(newpoint):
							line_disp, _ = obstacle.shortest_distance(newpoint)
							if (abs(line_disp.x) < abs(line_disp.y)):
								#displacement.x = -line_disp.x
								displacement.x = -0.2 * (abs(line_disp.x)/line_disp.x)
							else: 
								#displacement.y = -line_disp.y
								displacement.y = -0.2 * (abs(line_disp.y)/line_disp.y)
							boundary_flag = True; break
			points.append(newpoint)
		if boundary_flag: 
			points = [p+displacement for p in self.points]
		self.points = points

	def rotate(self, angle):
		"""
		Rotates the points in the shape and the shapes' image by the angle (rad)
		"""
		self.heading += angle
		polar = (int(self.rect.height/4), int(self.heading*180/pi+360)%360)
		rel_cor = pg.math.Vector2()
		rel_cor.from_polar(polar)
		cor = pg.math.Vector2(self.rect.center) + rel_cor # shift centre-of-rotation forward
		self.rotate_points(angle, tuple(cor))
		self.image = pg.transform.rotate(self.image_copy, (self.heading*-180)/(pi))

	def rotate_points(self, angle, center_of_rotation):
		"""
		Rotates self.points by 'angle' (rad) about the 'center_of_rotation' point
		@param int angle: Angle in radians to rotate
		@param tuple center_of_rotation: (x,y) point to rotate about
		"""
		center = pg.math.Vector2(center_of_rotation)
		for i, p in enumerate(self.points):
			p -= center
			original_x = p.x
			original_y = p.y
			p.x = original_x * cos(angle) - original_y * sin(angle)
			p.y = original_y * cos(angle) + original_x * sin(angle)
			p += center
			self.points[i] = p

	def update_checkpoints(self):
		"""
		Update the number of checkpoints the car has passed
		"""
		next_cp = self.checkpoints_passed % NUM_CHECKPOINTS
		#print(next_cp)
		if self.checkpoints[int(next_cp)].inside_polygon(self.rect.center):
			self.reward += 10
			self.checkpoints_passed += 1

	def get_gradual_accel(self):
		"""
		Returns a gradually increasing acceleration
		"""
		vel_percent = self.absolute_vel/self.max_vel
		if (vel_percent > 1): return self.d_accel
		grad_accel = 1 - (log(10-(9*(vel_percent)))/log(10))
		return (grad_accel*self.d_accel*(1-self.accel_coeff)+self.d_accel*self.accel_coeff)*(1+self.accel_coeff)

	def draw_polygon(self, screen, camera_offset, color=pg.Color("green"), width=2):
		"""Draw the shapes' polygon onto the game screen"""
		adjusted_pts = [p + pg.math.Vector2(camera_offset) for p in self.points]
		pg.draw.polygon(screen, color, adjusted_pts, width)

	def get_midpoint(self, start, end):
		"""
		Get midpoint of two Vector2() points
		"""
		x1, y1 = tuple(start)
		x2, y2 = tuple(end)
		return pg.math.Vector2((x1+x2)/2,(y1+y2)/2)	

	def laps_done(self):
		"""
		Number of laps completed by the car
		"""
		return int((self.checkpoints_passed)/NUM_CHECKPOINTS)

	def is_AI(self):
		"""
		Returns true if the class is an Car_AI class
		"""
		return self.AI

	def is_alive(self):
		"""
		Determine if the car has ever gone offroad
		"""
		return self.alive

	def kill(self):
		"""
		Sets self.alive to False to be used when the car 
		has gone offroad
		"""
		self.alive = False


class Car_AI(Car):
	"""
	Defines the functions used when the car is controlled by the NEAT algorithmn
	"""
	
	def __init__(self, point, degree, obstacles, checkpoints, walls, color=1):
		Car.__init__(self, point, degree, obstacles, checkpoints, color=color)
		self.lidar = LidarSensor(walls)
		self.turn_input = 0
		self.speed_input = 0
		self.AI = True

	def get_LIDAR(self):
		"""
		Gets car LIDAR readings. Gets the distances to the track edges
		"""
		return self.lidar.get_lidar_distances(self.rect.center , self.heading)

	def set_input(self, turn_input, speed_input):
		"""
		Sets the turning input of the car. Int between -1 to 1
		"""
		self.turn_input = turn_input
		if speed_input == 0: self.speed_input = -0.5
		else: self.speed_input = -1

	def update(self, offRoad = False):
		"""
		Updates the AI Car's sprite
		"""
		Car.update(self, offRoad = offRoad)

	def process_inputs(self):
		"""
		Gets input for the AI
		"""
		return self.speed_input, self.turn_input

	def get_reward(self):
		"""
		Returns the reward to train the NEAT algorithmn on
		"""
		reward = self.reward + 0.001*self.vel.magnitude()
		if abs(self.speed_input) == 1: reward += 0.005
		self.reward = 0
		return reward
	

class LidarSensor():
	"""
	Mimics the output of a lidar sensor mounted on the top of a car. Returns the 
	distance to obstacles at 45 deg intevals from the car's edges
	"""
	def __init__(self, obstacles):
		"""@param Obstacles list of Line()'s"""
		self.obstacles = obstacles
		self.center = (0,0)
		self.lidar_lines = []
		self.collisions = []

	def get_lidar_distances(self, center, headings):
		"""
		Executes get_lidar_lines and get_collision
		"""
		center = pg.math.Vector2(center)
		self.lidar_lines = self.get_lidar_lines(center, headings)
		distances = self.get_collisions(center, self.lidar_lines)
		return distances

	def get_lidar_lines(self, center, heading):
		"""
		Gets the lines that where the distance from the car to an obstacle
		will be determined.
		@param heading int (rad)
		@param center Vector2()
		"""
		lines = []
		cur_heading = heading * 180/pi
		new_line_headings = [0, 90, 45, -45]
		origin = pg.math.Vector2()
		for nh in new_line_headings:
			vec1 = origin
			if nh == 90:
				vec1.from_polar((250, nh+cur_heading))
				lines.append((center-vec1,center))
			elif nh == 45:
				vec1.from_polar((250, nh+cur_heading))
				lines.append((center-vec1,center))   
			elif nh == -45:	
				vec1.from_polar((250, nh+cur_heading))
				lines.append((center,vec1+center))
			elif nh == 0:
				vec1.from_polar((250, nh+cur_heading))
				lines.append((center-vec1,center))
				lines.append((center,vec1+center))
		return lines
			
	def get_collisions(self, center, lidar_line):
		"""
		@param lidar_line list of tuples of Vector2() of form (x,y)
		"""
		distances = []
		self.collisions = []
		for i, l in enumerate(lidar_line):
			line_cols = []
			for o in self.obstacles.items():    				
				collision_pt = o[1].is_line_collision(l[0], l[1])
				if collision_pt == None:
					col_flag = True
					line_cols.append(250)
				else:
					self.collisions.append(collision_pt)
					line_cols.append(int(center.distance_to(collision_pt)))
			distances.append(int(min(line_cols)))
		return distances

	def draw(self, screen, camera_offset, color=pg.Color("black"), width = 2):
		"""
		Draws the lines that make up self.points
		"""
		adjust = pg.math.Vector2(camera_offset)
		for c in self.lidar_lines:
			pt1, pt2 =  c[0] + adjust, c[1] + adjust
			pg.draw.line(screen, color, pt1, pt2, width)
		for c in self.collisions:
			pg.draw.circle(screen, pg.Color("red"), c + adjust, 8)
			

class Line():
	"""
	Class to define polygon & lines and what occurs when they are intersected
	"""
	
	def __init__(self, points):
		"""
		@param list points: List of tuple points defining a set of joined lines 
		"""
		self.points = points
		self.polygon = mplPath.Path(np.array([list(p) for p in points]))

	def inside_polygon(self, point):
		"""
		Detects if a point is inside the polygon.
		@param tuple point: (x,y)
		"""
		return self.polygon.contains_point(point)

	def shortest_distance(self,point):
		"""
		Returns the closest line from this class's polygon to the input point.
		@return line_points Vector2(): The 2 points that form the closest line
		@return displacement Vector2(): The relative displacement to that line 
		"""
		x, y = point
		distances, delta = [], []
		for i in range(-1, len(self.points)-1):
			x1,y1,x2,y2 = tuple(self.points[i])+tuple(self.points[i+1])
			A, B = x - x1, y - y1
			C, D = x2 - x1, y2 - y1
			dot = A * C + B * D
			len_sq = C * C + D * D
			param = -1
			if (len_sq != 0): param = dot
			if (param < 0): xx, yy = x1, y1
			elif (param > 1): xx, yy = x2, y2
			else: xx, yy = x1 + param * C, y1 + param * D
			dx, dy = x - xx, y - yy
			delta.append((dx, dy))
			distances.append(sqrt(dx * dx + dy * dy))
		min_dist = min(distances)
		min_index = distances.index(min_dist)
		line_points = self.points[min_index-1], self.points[min_index]
		displacement = pg.math.Vector2(delta[min_index])
		return displacement, line_points

	def is_point_collision(self, point):
		"""
		Detects if there is a collision between a point and the classes lines
		https://www.jeffreythompson.org/collision-detection/line-point.php
		@param Vector2() point
		"""
		for i in range(-1, len(self.points)-1):
			d1 = point.distance_to(self.points[i])
			d2 = point.distance_to(self.points[i+1])
			d3 = self.points[i].distance_to(self.points[i+1])
			result = abs(d3 - (d1+d2))
			if result < 0.08: return True 
		return False

	def is_line_collision(self, start, end):
		"""
		Detects if there is a collision between the input line and the lines 
		contained in this instance. Returns the first collision point.
		@param Vector2() start: First point of the input line
		@param Vector2() end: End point of the input line
		"""
		x3, y3, x4, y4 = tuple(start) + tuple(end)
		for i in range(-1, len(self.points)-1):
			x1, y1, x2, y2 = tuple(self.points[i]) + tuple(self.points[i+1])
			denomA = ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
			denomB = ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
			if denomA == 0 or denomB == 0: continue
			uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denomA
			uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denomB
			if (uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1): 
				intersection_pt = pg.math.Vector2(x1 + (uA * (x2-x1)), y1 + (uA * (y2-y1)))
				return intersection_pt
		return None

	def is_collision(self, shape_points):
		"""
		Detects if there is a collision with the input shape and this (self) line.
		@param list shape_points: List of Vector2() points making up the Shape()
		"""
		for i in range(-1, len(shape_points)-1):
			collision = self.is_line_collision(shape_points[i], shape_points[i+1])
			if collision != None: return collision
		return None

	def draw(self, screen, camera_offset, color=pg.Color("black"), width = 2):
		"""
		Draws the lines that make up self.points
		"""
		adjust = pg.math.Vector2(camera_offset)
		for i in range(-1, len(self.points)-1):
			pt1, pt2 =  self.points[i] + adjust, self.points[i+1] + adjust
			pg.draw.line(screen, color, pt1, pt2, width)


class Camera():
	"""
	Defines which areas of the map are shown on the pygame 
	screen at any time.
	"""
	def __init__(self, width, height):
		self.camera = pg.Rect(0, 0, width, height)
		self.width = width
		self.height = height

	def apply(self, entity):
		return entity.rect.move(self.camera.topleft)

	def apply_rect(self, rect):
		return rect.move(self.camera.topleft)

	def update(self, target):
		x = -target.rect.centerx + int(W_WIDTH / 2)
		y = -target.rect.centery + int(W_HEIGHT / 2)

		x, y = min(0, x), min(0, y)  # left and top scroll limit
		x = max(-(self.width - W_WIDTH), x)  # right
		y = max(-(self.height - W_HEIGHT), y)  # bottom
		self.camera = pg.Rect(x, y, self.width, self.height)
		return (x,y)


class TiledMap():
	"""
	Loads the tiled map and all of its elements
	"""
		
	def __init__(self, filename):
		self.tmxdata = tm = pytmx.load_pygame(filename, pixelalpha = True)
		self.width = tm.width * tm.tilewidth
		self.height = tm.height * tm.tileheight 
		self.map_scale_ratio = MAP_WIDTH / self.width

	def render(self, surface):
		"""
		Render all visible map layers onto the input pg.surface
		"""
		for layer in self.tmxdata.visible_layers:
			if isinstance(layer, pytmx.TiledTileLayer):
				for x, y, gid, in layer:
					tile = self.tmxdata.get_tile_image_by_gid(gid)
					if tile: 
						surface.blit(tile, 	(x*self.tmxdata.tilewidth, 
											y*self.tmxdata.tileheight,))

	def make_map(self):
		"""
		Forms the map by calling render()
		"""
		temp_surface = pg.Surface((self.width, self.height))
		self.render(temp_surface)
		temp_surface = scale_image(temp_surface, MAP_WIDTH)
		return temp_surface


##############################################
#############  Helper Functions  #############
##############################################

def update_rect(points):
	"""
	Updates the shapes' pg.rect using the shapes' points
	"""
	min_x, max_x = inf, -inf
	min_y, max_y = inf, -inf
	for point in points:
		min_x = min(min_x, point.x)
		max_x = max(max_x, point.x)
		min_y = min(min_y, point.y)
		max_y = max(max_y, point.y)
	width = max_x - min_x
	height = max_y - min_y
	return pg.Rect((min_x, min_y), (width, height))

def scale_image(image, width):
	"""
	Scales image to new width. Maintains ratio of width & height.
	"""
	cur_width, cur_heigh = image.get_size()
	height = int((cur_heigh/cur_width) * width)
	return pg.transform.scale(image, (width, height))
