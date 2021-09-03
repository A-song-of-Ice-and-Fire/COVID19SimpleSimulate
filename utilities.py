import math
import numpy as np
from typing import (
    Callable,
    Tuple ,
    Union ,
    Set
)

class ToolBox:
    def __init__(self) -> None:
        pass
    

    
    '''
    getSpecifiedCoo方法返回圆心为center、半径为r的圆中所有的整数坐标
    '''
    @staticmethod               
    def getSpecifiedCoo(center:Tuple[int,int],r:int)->Set:
        x_list = [x for x in range(math.ceil(center[0]-r),math.floor(center[0]+r+1))]
        coo_res = set([])
        for x in x_list:
            temp = math.sqrt(r**2-(x-center[0])**2)
            low , high = center[1]-temp , center[1]+temp
            for y in range(math.ceil(low),math.floor(high+1)):
                coo_res.add((x,y))
        return coo_res
    
    '''
    getBetaFunction方法返回函数getBeta，该函数输入距离，输出对应的感染概率
    '''
    @staticmethod
    def getBetaFunction(beta_init:float,critical_distance:Union[int,float])->Callable[[Union[int,float]],float]:
        def getBeta(distance):
            if distance<=critical_distance:
                return beta_init
            else:
                return beta_init * math.exp(critical_distance-distance)
        return getBeta
    
    '''
    rotation方法输入一个向量和旋转角度，返回旋转后的向量
    '''
    @staticmethod
    def rotation(vector:Tuple[int,int],angle:int)->Tuple[int,int]:
        rotation_mat = np.array((
            (math.cos(math.radians(angle)) , -math.sin(math.radians(angle))) , 
            (math.sin(math.radians(angle)) , math.cos(math.radians(angle))) 
            ))
        _vector = np.expand_dims(np.array(vector),-1)
        res_vector = np.matmul(rotation_mat,_vector).squeeze().tolist()
        return tuple([int(x) for x in res_vector])
    
    @staticmethod
    def getEuclideanDistance(coo_1:Tuple[int,int],coo_2:Tuple[int,int])->None:
        distance = math.sqrt( (coo_1[0]-coo_2[0]) ** 2 + (coo_1[1]-coo_2[1]) ** 2)
        return distance
