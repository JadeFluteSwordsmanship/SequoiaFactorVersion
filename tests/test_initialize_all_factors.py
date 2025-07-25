import sys
sys.path.append('.')  # 保证可以import到主目录下的包

def test_initialize_all_factors():
    from factors.utils import initialize_all_factors
    print("开始批量初始化所有注册因子（如数据量大，耗时较长）...")
    initialize_all_factors()
    print("所有因子初始化完毕！")

if __name__ == "__main__":
    test_initialize_all_factors()