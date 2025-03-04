import pandas as pd
from scipy.interpolate import interp1d


# 生成插值函数
def create_speed_functions(data):
    """
    创建每条道路的速度-时间插值函数。
    :param data: pandas DataFrame，包含 LinkID, Time, Speed 列
    :return: 字典 {LinkID: 插值函数}
    """
    speed_functions = {}
    for link_id in data['LinkID'].unique():
        road_data = data[data['LinkID'] == link_id]  # 获取特定 LinkID 的数据
        time = road_data['Time'].values  # 时间点
        speed = road_data['Speed'].values  # 对应速度

        # 创建插值函数
        f = interp1d(time, speed, kind='nearest', fill_value="extrapolate")
        speed_functions[link_id] = f

    return speed_functions


# 查询函数
def get_speed(speed_functions, link_id, time):
    """
    查询给定 LinkID 和时间的车速。
    :param speed_functions: 速度-时间插值函数
    :param link_id: 道路ID
    :param time: 时间（单位：分钟）
    :return: 对应时间的车速
    """
    if link_id not in speed_functions:
        raise ValueError(f"LinkID {link_id} 不存在")

    # 调用对应的插值函数
    speed = speed_functions[link_id](time)
    return speed


if __name__ == "__main__":

    # 读取 CSV 文件
    file_path = "linkID_time_speed_12-7.csv"
    data = pd.read_csv(file_path)

    # 创建速度插值函数
    speed_functions = create_speed_functions(data)

    # 示例：查询某条道路在指定时间的车速
    link_id = 14  # 替换为实际道路ID
    time = 0  # 替换为实际时间（分钟）
    try:
        speed = get_speed(speed_functions, link_id, time)
        print(f"LinkID {link_id} 在时间 {time} 分钟的车速为: {speed:.2f}")
    except ValueError as e:
        print(e)
