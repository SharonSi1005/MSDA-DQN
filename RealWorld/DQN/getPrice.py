import pandas as pd

# 读取 CSV 文件
file_path = 'charging_prices_processed.csv'  # 替换为实际文件路径
data = pd.read_csv(file_path)


def get_price(charging_id, time_min):
    """
    查询给定充电桩ID和时间（分钟）对应的电价。
    :param charging_id: 充电桩ID
    :param time_min: 时间（以分钟为单位）
    :return: 对应时间的电价
    """
    # 筛选指定 ID 的数据
    filtered_data = data[data['ID'] == charging_id]
    if filtered_data.empty:
        raise ValueError(f"充电桩ID {charging_id} 不存在")

    # 查找对应时间段
    row = filtered_data[(filtered_data['StartTime'] <= time_min) &
                        (filtered_data['EndTime'] >= time_min)]
    if row.empty:
        raise ValueError(f"时间 {time_min} 不在充电桩ID {charging_id} 的任何时间段中")

    # 返回电价
    return row['Price'].values[0]


if __name__ == "__main__":
    # 示例：查询某充电桩的电价
    charging_id = 13449  # 替换为实际ID
    time_min = 0  # 替换为实际时间（分钟）
    try:
        price = get_price(charging_id, time_min)
        print(f"充电桩ID {charging_id} 在时间 {time_min} 分钟的电价为: {price:.2f}")
    except ValueError as e:
        print(e)
