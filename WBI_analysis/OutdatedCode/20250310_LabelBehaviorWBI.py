import pandas as pd
import time
import os
from pynput import keyboard


# 保存文件地址和文件名
save_path = r'D:\data analysis\demo data'
file_name = '20250312_0g-24d-4h_1'

# 初始化 DataFrame
columns = ["Time", "Forward", "Turn", "Reverse"]
behav_keys = ['f', 't', 'r']

df = pd.DataFrame(columns=columns)

# 监听键盘输入ft
pressed_key = None  # 记录按下的键
exit_flag = False   # 退出标志

def on_press(key):
    global pressed_key, exit_flag
    try:
        if hasattr(key, 'char') and key.char in behav_keys:
            pressed_key = key.char.upper()  # 记录按键（F/T/R）
        elif key == keyboard.Key.enter:
            print("Exit command detected. Saving data and exiting...")
            exit_flag = True
            return False  #
            # 停止监听
    except AttributeError:
        print('Error input, not a key in behavioral list')
        pass  # 处理特殊按键


def log_keypress(interval=1):
    global df, pressed_key

    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    row = {
        "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Forward": 0,
        "Turn": 0,
        "Reverse": 0
    }

    while not exit_flag:  # 监听键盘的同时循环

        # current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        current_time = pd.Timestamp.now()
        # 读取键盘状态
        if pressed_key:
            row = {
                "Time": current_time,
                "Forward": int(pressed_key == 'F'),
                "Turn": int(pressed_key == 'T'),
                "Reverse": int(pressed_key == 'R')
            }
        else:
            # 如果键盘没有输入
            row['Time'] = current_time

        pressed_key = None  # 重置按键状态

        # 追加数据
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print(f"Logged: {row}")  # 调试信息

        time.sleep(interval)  # 每 interval 秒循环一次

    listener.stop()
    # 保存 CSV
    df.to_csv(os.path.join(save_path, file_name+'.csv'), index=False)
    print("Data saved to key_log.csv")


# 运行脚本
if __name__ == "__main__":
    log_keypress()
