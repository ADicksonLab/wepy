import random as rand


from wepy.runner import Runner
from wepy.walker import Walker


class RandomWalkRunner(Runner):
    
    def __init__(self, dimension=2, probability=0.25):
        """Initialize RandomWalk object with the number of 
        dimension and probability"""
        self.dimension = dimension
        self.probability = probability
        
    
    def walk(self, positions):
        new_positions = positions
        for dimension in range(self.dimension):
                r = rand.uniform(0,1)
                 # produce  a random number to chose beween increasing or decreasing position
                 
                if r < self.probability:
                    new_positions[dimension] += 1
                     
                else:
                    new_positions[dimension] -=1

                # implement the boundary condition for movement, movement to -1 are rejected
                if new_positions[dimension] < 0:
                    new_positions[dimension] = 0
                    
        return new_positions
               
    def run_segment(self, walker, segment_length):
        positions = walker.positions
    
        for segment_idx in range(segment_length):
            new_positions = self.walk(positions)
            positions = new_positions

        new_state = State(new_positions, 0.0)
        new_walker = RandomWalker(new_state, walker.weight)    
 #       print (new_walker.state.positions) 
        return new_walker


class RandomWalker(Walker):
    
    def __init__(self, state, weight):
        super().__init__(state, weight)        
        self._keys = ['positions', 'time']

    def dict(self):
        """return a dict of the values."""
        return {'positions' : self.state.positions(),
                'time' : self.time,
               }
    def keys(self):
        return self._keys

    def __getitem__(self, key):
        if key == 'positions':
            return self.state.positions()
        elif key == 'time':
            return self.time
        else:
            raise KeyError('{} not an RandomWalker attribute')

    
    @property
    def positions(self):
        try:
            return self.state.getPositions()
        except TypeError:
            return None

    @property
    def positions_unit(self):
        return self.positions.unit
    @property
    def time(self):
        return self.state.getTime()
  
    
        
    
class State(object):
    def __init__(self, positions, time):
        self.positions = positions.copy()
        self.time = time 
    # @property
    # def positions(self):
    #      return self.positions

    # @property
    # def time(self):
    #      return self.time

    def getPositions(self):
        return self.positions

    def getTime(self):
        return self.time
    

    
        
        
