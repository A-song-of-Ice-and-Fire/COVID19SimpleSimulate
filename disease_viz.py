from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from disease_model import *
from utilities import ToolBox
def agent_portrayal(agent):
    if not agent.pos:
        return None
    
    
    portrayal = DiseaseModel.getBasePortrayal(agent)
    if isinstance(agent,Individual) and agent.status == Status.I:      #只支持圆和正方形
            portrayal["Color"] = "red"
    
    return portrayal
if __name__ == "__main__":
    space_viz = CanvasGrid(agent_portrayal,100,100,800,800)   #最大为400x400
    chart_viz = ChartModule(
        [{"Label" : "s_ratio" , "Color" : "green"} , {"Label" : "i_ratio" , "Color" : "red"}] ,
        data_collector_name="data_collector"
    )
    
    getBeta = ToolBox.getBetaFunction(0.5,2)      #基础感染概率为0.5，阈值距离为2
    
    server = ModularServer(
        DiseaseModel, 
        [space_viz,chart_viz] ,                       #可视化对象列表
        "Money Model" ,
        {
            "individual_nums" : [10,30] ,
            "init_I_nums" : [3,6], 
            "grid_size" : (100,100),
            "infect_scope":3,
            "getBeta":getBeta,                  #获取感染率的函数，戴口罩可使得基础感染率下降
            "min_contact_distance" : 3          #限制距离
            }
        )                   #这样绘制图像 轨迹不会消失
    server.port = 8521
    server.launch()