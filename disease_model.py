from asyncio.windows_events import NULL
from mesa import Agent , Model
from enum import Enum, unique
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from rewrite_method import model_methods
from utilities import ToolBox
from typing import (
    Tuple , 
    List ,
)
import math
import numpy as np
def sign(x):
    if x>0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


class Status(Enum):
    S = 0
    I = 1

def compute_i_ratio(model):
    agent_nums = model.individual_nums
    i_nums= 0
    for agent in model.schedule.agents:
        if agent.status == Status.I:
            i_nums += 1
    return i_nums / agent_nums

def compute_s_ratio(model):
    agent_nums = model.individual_nums
    s_nums= 0
    for agent in model.schedule.agents:
        if agent.status == Status.S:
            s_nums += 1

    return s_nums / agent_nums

class Brick(Agent):
    def __init__(self,unique_id,model):
        super().__init__(unique_id,model)
        self.model = model
        self.pos = None
    def step(self):
        pass

class Architecture():
    def __init__(self,id_counter,model,pos:tuple,width=1,height=1):
        self.width , self.height = width ,height
        self.model = model
        self.bricks = []
        self.layer = 0
        self.give_positions(pos,id_counter)

    
    def give_positions(self,pos:tuple,id_counter):
        def is_pos_available(pos):
            for dx in range(0,self.width):
                for dy in range(0,self.height):
                    if not self.model.grid.is_cell_empty((pos[0]+dx , pos[1]+dy)):
                        return False
            return True

        for _ in range(20):
            if is_pos_available(pos):
                self.pos = pos
                for dx in range(0,self.width):
                    for dy in range(0,self.height):
                        brick = Brick(next(id_counter),self.model)
                        self.bricks.append(brick)
                        self.model.schedule.agents.append(brick)
                        self.model.grid.place_agent(brick,(self.pos[0]+dx,self.pos[1]+dy))
                return
            else:
                pos = self.model.grid.find_empty()
        raise Exception("grid中没有空的位置分配给新的brick")

class Individual(Agent):
    def __init__(self,unique_id,model):
        super().__init__(unique_id,model)
        self.status = Status.S
        self.pos = None
    
    #移动方式:随机移动，一个格子里至多只能有一个agent
    def move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos ,
            moore = True ,
            include_center = False
        )

        empty_neighbors = []
        for neighbor in neighbors:
            if self.model.grid.is_cell_empty(neighbor):
                empty_neighbors.append(neighbor)

        can_be_placed = []
        if self.model.min_contact_distance > 0:
            #扫描警戒范围内是否存在Individual
            specifiedCoo = ToolBox.getSpecifiedCoo(self.pos,self.model.min_contact_distance)
            alert_distances = {}
            for coo in specifiedCoo:
                #若为循环边界，则对其进行转化
                if self.model.grid.torus:
                    coo = self.model.grid.torus_adj(coo)
                    


                if not self.model.grid.is_cell_empty(coo) and isinstance(self.model.grid[coo[0]][coo[1]],Individual):
                    alert_distances[coo] = ToolBox.getEuclideanDistance(self.pos,coo)
            #在moore邻居中寻找可移动的邻居
            if len(alert_distances) > 0 :
                for neighbor in empty_neighbors:
                    for object_coo , object_distance in alert_distances.items():
                        if object_distance > ToolBox.getEuclideanDistance(neighbor,object_coo):
                            break
                    else:
                        can_be_placed.append(neighbor)
        can_be_placed = can_be_placed if len(can_be_placed)>0 else empty_neighbors

        if can_be_placed:
            new_pos = self.random.choice(can_be_placed)
            self.model.grid.move_agent(self,new_pos)
    
    def spread(self): #向附近格子和当前
        if self.status == Status.I:
            #neighbors_pos = self.model.grid.get_ring_neighborhood(self.model.grid,pos=self.pos,radius=self.model.infect_scope)
            #neighbors = self.model.grid.get_cell_list_contents(neighbors_pos)
            neighbors = self.model.grid.get_neighbors(pos=self.pos,moore=True,radius = self.model.infect_scope)
            if len(neighbors)>0 :
                for neighbor in neighbors:
                    if not isinstance(neighbor,Brick):
                        distance = self.relativeDistance(neighbor) 
                        generated_number = self.random.random()
                        if generated_number < self.model.getBeta(distance) and neighbor.status == Status.S:
                            neighbor.status = Status.I
                        
    def relativeDistance(self,other:Agent)->float or None:
        if (other.pos is None) or (self.pos is None):
            return None
        return ToolBox.getEuclideanDistance(self.pos,other.pos)

    def step(self):
        if self.pos is not None:
            self.spread()
            self.move()


class Audience(Individual):
    def __init__(self,unique_id,model):
        super().__init__(unique_id,model)
        self.status = Status.S
        self.pos = None
        self.speed_set = set([(1,0),(0,1),(-1,0),(0,-1)])
        self.last_v = self.random.sample(self.speed_set,1)[0]
    #移动方式:随机移动，一个格子里至多只能有一个agent
    def move(self,last_v):
        '''
        计算当前速度，行为模式为：
            若上步的速度矢量与距离矢量夹角小于90度，则速度矢量不变
            若上步的速度矢量与距离矢量夹角为90度，则速度矢量旋转至于距离矢量同向
            若上步的速度矢量与距离矢量夹角大于90度，则速度矢量反向
        注意，速度矢量只可能为Union((1,0),(0,1),(-1,0),(0,-1))
        
        '''
        assert last_v in self.speed_set  , '速度矢量不正确'
        if self.model.end_point.pos in self.model.grid.get_neighborhood(self.pos,False):
            v = (self.model.end_point.pos[0] - self.pos[0],self.model.end_point.pos[1] - self.pos[1])
            self.model.grid.remove_agent(self)
        else:
            distance = (self.model.end_point.pos[0] - self.pos[0],self.model.end_point.pos[1] - self.pos[1])
            dot_res = np.dot(np.array(distance),np.array(last_v))
            angle = 0
            if dot_res == 0:
                angle = 90 * sign(np.cross(np.array(last_v),np.array(distance)))    #利用二维叉乘判定速度向量和距离向量的相对角度关系（顺逆时针关系）
            elif dot_res < 0:
                angle = 180
            v = ToolBox.rotation(last_v,angle)

            new_pos = (self.pos[0]+v[0],self.pos[1]+v[1])

            next_content = self.model.grid[new_pos[0]][new_pos[1]]
            if isinstance(next_content,Individual):
                v = last_v
            elif isinstance(next_content,Brick):
                pass
            else:
                    self.model.grid.move_agent(self,new_pos)
        return v


        '''          
        neighbors = self.model.grid.get_neighborhood(
            self.pos ,
            moore = True ,
            include_center = False
        )

        empty_neighbors = []
        for neighbor in neighbors:
            if self.model.grid.is_cell_empty(neighbor):
                empty_neighbors.append(neighbor)

        if len(empty_neighbors) > 0:
            new_pos = self.random.choice(empty_neighbors)
            self.model.grid.move_agent(self,new_pos)
        '''
    def step(self):
        if self.pos is not None:
            self.spread()
            self.last_v = self.move(self.last_v)


class DiseaseModel(Model):
    def __init__(self,individual_nums,init_I_nums,grid_size:tuple,getBeta=0.5,infect_scope=3,min_contact_distance = 0):
        '''
        individual_nums : 一个list，list[0]为观赛者的数量 , list[1]为普通人的数量
        '''
        self.individual_nums = sum(individual_nums)
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(*grid_size,True)
        self.running = True
        self.getBeta = getBeta
        self.min_contact_distance = min_contact_distance
        self.infect_scope = infect_scope 
        self.id_counter = self.id_count()
        end_point_coo = (self.random.randrange(self.grid.width),self.random.randrange(self.grid.height))
        self.end_point = Architecture(self.id_counter,self,end_point_coo,1,1)

        infected_ids = self.random.sample(range(self.id_counter.send(True),self.id_counter.send(True)+individual_nums[0]),init_I_nums[0]) 
        infected_ids += self.random.sample(range(self.id_counter.send(True)+individual_nums[0],self.id_counter.send(True)+individual_nums[0]+individual_nums[1]),init_I_nums[1])

        self.infected_ids = sorted(infected_ids) 
        infected_index = 0


        for _ in range(individual_nums[0]):     #创建观赛者
            agent = Audience(next(self.id_counter),self)
            if infected_index<init_I_nums[0] and self.infected_ids[infected_index] == agent.unique_id:
                agent.status = Status.I
                infected_index += 1
            self.schedule.add(agent)
            self.grid.position_agent(agent)
            
        for _ in range(individual_nums[1]):     #创建普通人
            agent = Individual(next(self.id_counter),self)
            if infected_index<(init_I_nums[0]+init_I_nums[1]) and self.infected_ids[infected_index] == agent.unique_id:
                agent.status = Status.I
                infected_index += 1
            self.schedule.add(agent)
            self.grid.position_agent(agent)
        self.data_collector = DataCollector(
            model_reporters = {"s_ratio" : compute_s_ratio , "i_ratio" : compute_i_ratio}
        )
        for agent in self.schedule.agents:
            print(f"{agent.status}")
        for method in model_methods: 
            exec(f"self.grid.{method.__name__}=method")
            
    def step(self):
        self.schedule.step()
        self.data_collector.collect(self)


    def id_count(self):
        cur_id , max_id = 0 , self.grid.width * self.grid.height
        while cur_id < max_id:
            status = yield cur_id
            if not status:
                cur_id += 1
        raise StopIteration("已达最大编号")
    
    @staticmethod
    def getBasePortrayal(Agent):
        portrayal_brick = {
                "Shape" : "rect",
                "Color" : "black",
                "Filled" : "true",
                "Layer" : 0,            #图层 越大则在越上面
                "w"     : 1,           #绘制圆的直径，一般一个格子的边长为1
                "h"     : 1
            }
        portrayal_individual = {
                "Shape" : "circle",
                "Color" : "blue",
                "Filled" : "true",
                "Layer" : 0,            #图层 越大则在越上面
                "r"     : 0.5
            }
        portrayal_audience = {
            "Shape" : "rect",
            "Color" : "blue",
            "Filled" : "true",
            "Layer" : 0,            #图层 越大则在越上面
            "w"     : 0.5,           #绘制圆的直径，一般一个格子的边长为1
            "h"     : 0.5
        }
        if isinstance(Agent,Brick):
            return portrayal_brick
        elif isinstance(Agent,Audience):
            return portrayal_audience
        else:
            return portrayal_individual


if __name__ == "__main__":
    beta = DiseaseModel.getBetaFunction(0.5,2)
    print(beta(4))