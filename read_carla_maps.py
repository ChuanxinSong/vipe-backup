import carla

try:
    # 连接到正在运行的 CARLA 服务器
    client = carla.Client('localhost', 12369)
    client.set_timeout(5.0) # 设置5秒超时

    # 获取可用的地图列表
    available_maps = client.get_available_maps()

    print("✅ CARLA 服务器连接成功！")
    print("-----------------------------------------")
    print("🗺️ 在此镜像中可用的地图列表:")

    # 打印出所有地图的路径
    for map_path in sorted(available_maps):
        print(f"  -> {map_path}")

    print("-----------------------------------------")

except Exception as e:
    print(f"❌ 连接 CARLA 服务器失败: {e}")