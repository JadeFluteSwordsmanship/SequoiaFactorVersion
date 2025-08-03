import sys
sys.path.append('.')  # 保证可以import到主目录下的包

def test_initialize_all_factors():
    from factors.utils import initialize_all_factors
    import time
    
    print("开始批量初始化所有注册因子（优化版本：数据聚类+并行处理）...")
    
    start_time = time.time()
    
    # 使用4个并行进程，强制重新计算
    initialize_all_factors(force=False, max_workers=11)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"所有因子初始化完毕！总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    test_initialize_all_factors()