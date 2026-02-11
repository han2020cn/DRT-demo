pip install -r requirements.txt
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import random
from shapely.geometry import LineString
#3
import folium
from folium.plugins import AntPath
#4
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- 辅助函数：将节点ID转换为经纬度 ---
def node_to_latlon(G_4326, node_id):
    if node_id not in G_4326.nodes:
        return None
    return (G_4326.nodes[node_id]['y'], G_4326.nodes[node_id]['x'])

"""负责蒙特卡洛生成乘客需求"""
class DemandManager:
    def __init__(self, G, rate_per_hour, hours):
        self.G = G
        self.total_demand = rate_per_hour * hours
        self.duration_sec = hours * 3600
        
    def generate_requests(self):
        nodes = list(self.G.nodes)
        # 蒙特卡洛生成下单时间
        times = np.sort(np.random.uniform(0, self.duration_sec, self.total_demand))
        requests = []
        for i, t in enumerate(times):
            origin, dest = random.sample(nodes, 2)
            requests.append({
                'id': i, 'time': t, 
                'origin': origin, 'dest': dest,
                'status': 'waiting'
            })
        return requests

"""负责车辆状态、路径规划和移动"""
class SimpleVehicle:
    def __init__(self, vehicle_id, start_node, speed_kmh=30):
        self.id = vehicle_id
        self.pos = start_node
        self.speed = speed_kmh / 3.6  # m/s
        self.stop_time_sec = stop_time_sec # 停车上下客时间
        self.current_path = [] # 车辆当前行驶的节点序列
        self.path_idx = 0 # 在当前路径中的索引
        self.trajectory_records = [] # 记录每秒的 (lat, lon, status, active_requests_info)
        
        self.passengers_on_board = [] # 车上乘客请求ID
        self.pending_requests = [] # 等待被服务的请求ID
        self.completed_requests = [] # 已完成的请求ID
    def add_request(self, request):
        self.pending_requests.append(request)

    def update_trajectory(self, G_proj, G_4326, current_simulation_time, status, request_data_for_this_frame):
        """记录当前车辆位置、状态和活跃请求信息"""
        latlon = node_to_latlon(G_4326, self.pos)
        
        # 记录当前待处理和在途的请求信息
        active_requests_info = []
        for req in self.pending_requests:
            active_requests_info.append({
                'id': req['id'],
                'origin': node_to_latlon(G_4326, req['origin']),
                'dest': node_to_latlon(G_4326, req['dest']),
                'type': 'pending_pickup'
            })
        for req_id in self.passengers_on_board:
             # 假设self.passengers_on_board里存的是原始request对象
            req_obj = next((r for r in self.pending_requests if r['id'] == req_id), None)
            if req_obj:
                active_requests_info.append({
                    'id': req_obj['id'],
                    'origin': node_to_latlon(G_4326, req_obj['origin']), # 乘客已上车，这里依然显示其起点，但状态不同
                    'dest': node_to_latlon(G_4326, req_obj['dest']),
                    'type': 'on_board'
                })
                
        self.trajectory_records.append({
            'time': current_simulation_time,
            'lat': latlon[0],
            'lon': latlon[1],
            'status': status,
            'active_requests': active_requests_info # 包含请求信息
        })

    def drive_path(self, G_proj, G_4326, path_nodes, start_sim_time, current_request_id=None, task_type='drive'):
        """
        根据路径节点模拟车辆行驶，并记录每秒轨迹。
        task_type: 'to_pickup', 'to_dropoff', 'idle'
        """
        current_path_coords = []
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_data = G_proj.get_edge_data(u, v)[0]
            length = edge_data['length']
            drive_duration = length / self.speed # 这段路行驶时间
            
            # 插值点
            start_coord = node_to_latlon(G_4326, u)
            end_coord = node_to_latlon(G_4326, v)
            
            num_steps = max(1, int(drive_duration)) # 至少一步
            
            for step in range(num_steps):
                current_sim_time = start_sim_time + len(self.trajectory_records)
                alpha = step / num_steps
                interp_lat = start_coord[0] + (end_coord[0] - start_coord[0]) * alpha
                interp_lon = start_coord[1] + (end_coord[1] - start_coord[1]) * alpha
                
                # 记录车辆轨迹
                self.trajectory_records.append({
                    'time': current_sim_time,
                    'lat': interp_lat,
                    'lon': interp_lon,
                    'status': task_type,
                    'passengers_on_board': list(self.passengers_on_board), # 记录车上乘客ID
                    'pending_requests': [r['id'] for r in self.pending_requests if r['status'] == 'pending'] # 记录等待乘客ID
                })
        
        self.pos = path_nodes[-1] # 更新车辆当前所在节点
        return start_sim_time + len(self.trajectory_records) # 返回本次行驶结束的模拟时间

    def simulate_stop(self, G_4326, start_sim_time, stop_type='pickup_stop'):
        """模拟车辆停靠时间"""
        for _ in range(self.stop_time_sec):
            current_sim_time = start_sim_time + len(self.trajectory_records)
            latlon = node_to_latlon(G_4326, self.pos)
            self.trajectory_records.append({
                'time': current_sim_time,
                'lat': latlon[0],
                'lon': latlon[1],
                'status': stop_type,
                'passengers_on_board': list(self.passengers_on_board),
                'pending_requests': [r['id'] for r in self.pending_requests if r['status'] == 'pending']
            })
        return start_sim_time + len(self.trajectory_records)


# --- 运行仿真循环 ---
# 1. 初始化地图和组件
G_proj = ox.project_graph(ox.graph_from_address("Loughborough, Leicestershire, UK", dist=1500, network_type='drive'))

largest_cc = max(nx.strongly_connected_components(G_proj), key=len)
G_proj = G_proj.subgraph(largest_cc).copy()

dm = DemandManager(G_proj, rate_per_hour=10, hours=2)
requests = dm.generate_requests()
car = SimpleVehicle(1, start_node=list(G_proj.nodes)[0])
current_sim_time = 0

# 存储一个请求的原始状态，方便后续查找
request_lookup = {req['id']: req for req in all_requests}

# 遍历所有请求（FIFO）
for req_id in sorted(request_lookup.keys()): # 确保按时间顺序处理
    req = request_lookup[req_id]

    # 等待请求发生
    if req['time'] > current_sim_time:
        # 如果车辆空闲，记录空闲轨迹
        idle_duration = int(req['time'] - current_sim_time)
        if idle_duration > 0:
            for _ in range(idle_duration):
                car.trajectory_records.append({
                    'time': current_sim_time + len(car.trajectory_records),
                    'lat': node_to_latlon(G_4326, car.pos)[0],
                    'lon': node_to_latlon(G_4326, car.pos)[1],
                    'status': 'idle',
                    'passengers_on_board': list(car.passengers_on_board),
                    'pending_requests': [r_['id'] for r_ in all_requests if r_['status'] == 'pending' and r_['time'] <= current_sim_time + len(car.trajectory_records)] # 包含所有已发生的请求
                })
        current_sim_time = req['time'] # 仿真时间跳到请求发生时

    # 将请求加入待处理列表（这里是FIFO，所以每次只处理一个）
    req['status'] = 'pending'
    car.add_request(req) # 假设只加这一个请求

    # --- A. 前往接人 ---
    path_to_p = nx.shortest_path(G_proj, car.pos, req['origin'], weight='length')
    current_sim_time = car.drive_path(G_proj, G_4326, path_to_p, current_sim_time, task_type='to_pickup')
    current_sim_time = car.simulate_stop(G_4326, current_sim_time, stop_type='pickup_stop')

    # 接到乘客
    car.passengers_on_board.append(req['id'])
    req['status'] = 'picking_up' # 乘客已上车，状态变为上车中
    req['pickup_time'] = current_sim_time # 记录接人时间
    car.pending_requests.remove(req) # 从待处理中移除

    # --- B. 送人到终点 ---
    path_to_d = nx.shortest_path(G_proj, car.pos, req['dest'], weight='length')
    current_sim_time = car.drive_path(G_proj, G_4326, path_to_d, current_sim_time, task_type='to_dropoff')
    current_sim_time = car.simulate_stop(G_4326, current_sim_time, stop_type='dropoff_stop')

    # 乘客下车
    car.passengers_on_board.remove(req['id'])
    req['status'] = 'completed'
    req['dropoff_time'] = current_sim_time # 记录下车时间
    car.completed_requests.append(req['id']) # 加入已完成列表

'''# 3. 输出车辆轨迹
df_trajectory = pd.DataFrame(car.trajectory, columns=['latitude', 'longitude', 'status'])
df_trajectory.to_csv('vehicle_trajectory.csv', index=False)'''

# 3. 输出车辆轨迹
import folium
from folium.plugins import TimestampedGeoJson
import datetime

def create_folium_dynamic_map(G_proj, trajectory_records, all_requests, filename="folium_simulation.html"):
    # 1. 初始化地图
    # 投影回经纬度获取中心点
    G_4326 = ox.projection.project_graph(G_proj, to_crs='EPSG:4326')
    avg_lat = np.mean([data['y'] for node, data in G_4326.nodes(data=True)])
    avg_lon = np.mean([data['x'] for node, data in G_4326.nodes(data=True)])
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles='CartoDB dark_matter')

    # 2. 准备时间序列数据 (GeoJSON Features)
    features = []
    
    # 仿真起始时间设定为 2026-01-21 09:00:00 (符合运营时间设定)
    start_dt = datetime.datetime(2026, 1, 21, 9, 0, 0)

    # --- A. 添加车辆位置点 (随时间移动) ---
    # 为了减少HTML体积，每隔 30 秒记录一个位置点
    for i in range(0, len(trajectory_records), 30):
        rec = trajectory_records[i]
        timestamp = (start_dt + datetime.timedelta(seconds=rec['time'])).isoformat()
        
        color = '#FFFF00' # Idle
        if rec['status'] == 'to_pickup': color = '#FFA500' # Orange
        elif rec['status'] == 'to_dropoff': color = '#00FF00' # Lime
        
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [rec['lon'], rec['lat']],
            },
            'properties': {
                'time': timestamp,
                'style': {'color': color, 'fillColor': color},
                'icon': 'circle',
                'iconstyle': {'fillOpacity': 1, 'radius': 5},
                'popup': f"Status: {rec['status']}"
            }
        })

    # --- B. 添加需求点配对 (从下单时刻出现，到送达时刻消失) ---
    for req in all_requests:
        # 下单时间
        req_time = (start_dt + datetime.timedelta(seconds=req['time'])).isoformat()
        # 消失时间 (如果已完成则为下车时间，否则为仿真结束)
        end_time_sec = req['dropoff_time'] if req['dropoff_time'] > 0 else trajectory_records[-1]['time']
        end_time = (start_dt + datetime.timedelta(seconds=end_time_sec)).isoformat()
        
        origin = node_to_latlon(G_4326, req['origin'])
        dest = node_to_latlon(G_4326, req['dest'])

        # 绘制起点 (圆点)
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [origin[1], origin[0]]},
            'properties': {
                'time': req_time,
                'endtime': end_time,
                'style': {'color': 'red', 'fillColor': 'red'},
                'iconstyle': {'radius': 4, 'fillOpacity': 0.8},
                'popup': f"Request {req['id']} Origin"
            }
        })

        # 绘制终点 (叉号/小方块)
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [dest[1], dest[0]]},
            'properties': {
                'time': req_time,
                'endtime': end_time,
                'style': {'color': 'pink', 'fillColor': 'pink'},
                'iconstyle': {'radius': 3, 'fillOpacity': 0.5},
                'popup': f"Request {req['id']} Destination"
            }
        })

        # 绘制连接起终点的虚线 (体现配对)
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [[origin[1], origin[0]], [dest[1], dest[0]]]
            },
            'properties': {
                'time': req_time,
                'endtime': end_time,
                'style': {'color': 'red', 'weight': 1, 'dashArray': '5, 5', 'opacity': 0.4}
            }
        })

    # 3. 添加 TimestampedGeoJson 插件
    TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='PT30S', # 步长30秒
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=10,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True
    ).add_to(m)

    m.save(filename)
    print(f"Folium 交互仿真地图已保存为: {filename}")

# 使用之前代码生成的变量
