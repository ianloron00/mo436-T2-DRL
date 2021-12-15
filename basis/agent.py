from basis.dependencies import *

class Agent:
  def __init__(self, x=np.random.randint(0,10), y=np.random.randint(0,10)):
    self.x = x
    self.y = y
  
  def __sub__(self, other):
    return (self.x - other.x, self.y - other.y)
  
  def __eq__(self, o: object) -> bool:
      return (self.x == o.x and self.y == o.y)

  def __str__(self):
    return f"{self.x}, {self.y}"
  
  def choose_action(self, env):
    action = env.action_space.sample()
    return action
  
  def set_position(self, x, y):
    self.x = x
    self.y = y
  
  def get_position(self):
    return (self.x, self.y)

  def can_move(self, env, x, y):
    return False if (x < 0 or x > env.width-1 or y < 0 or y > env.height-1) else True
  
  @staticmethod
  def get_direction(choice):
    if choice == 0:
        return (1, 0)
    elif choice == 1:
        return (-1,0)
    elif choice == 2:
        return (0, 1)
    else:
        return (0,-1)

  def action(self, env, choice):
    x, y = self.get_direction(choice)
    return self.move(env, x, y)

  def move(self, env, x=-1000, y=-1000):

    if x == -1000:
        self.x += np.random.randint(-1, 2)
    else:
        self.x += x
    
    if y == -1000:
        self.y += np.random.randint(-1, 2)
    else:
        self.y += y

    moved = self.can_move(env, self.x, self.y)

    if not moved:
        if self.x < 0:
            self.x = 0
        elif self.x > env.width-1:
            self.x = env.width-1

        if self.y < 0:
            self.y = 0
        elif self.y > env.height-1:
            self.y = env.height-1

    return moved