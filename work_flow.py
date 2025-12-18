# -*- encoding: UTF-8 -*-

import data_fetcher
import settings
import strategy.enter as enter
from strategy import turtle_trade, climax_limitdown
from strategy import backtrace_ma250
from strategy import breakthrough_platform
from strategy import parking_apron
from strategy import low_backtrace_increase
from strategy import keep_increasing
from strategy import high_tight_flag
import akshare as ak
import data_reader
import push
import logging
import time
import datetime
import daily_data_fetcher
from factors.utils import update_all_factors_daily
from datetime import datetime
import push

num_map = {
    1: '首', 2: '二', 3: '三', 4: '四', 5: '五', 
    6: '六', 7: '七', 8: '八', 9: '九', 10: '十',
    11: '十一', 12: '十二', 13: '十三'
}

def prepare_old():
    logging.info("************************ process start ***************************************")
    all_data = ak.stock_zh_a_spot_em()
    subset = all_data[['代码', '名称']]
    stocks = [tuple(x) for x in subset.values]
    statistics(all_data, stocks)

    strategies = {
        '放量上涨': enter.check_volume,
        '均线多头': keep_increasing.check,
        '停机坪': parking_apron.check,
        '回踩年线': backtrace_ma250.check,
        # '突破平台': breakthrough_platform.check,
        '无大幅回撤': low_backtrace_increase.check,
        '海龟交易法则': turtle_trade.check_enter,
        '高而窄的旗形': high_tight_flag.check,
        '放量跌停': climax_limitdown.check,
    }

    if datetime.datetime.now().weekday() == 0:
        strategies['均线多头'] = keep_increasing.check

    process(stocks, strategies)


    logging.info("************************ process   end ***************************************")

def process(stocks, strategies):
    stocks_data = data_fetcher.run(stocks)
    for strategy, strategy_func in strategies.items():
        check(stocks_data, strategy, strategy_func)
        time.sleep(2)

def check(stocks_data, strategy, strategy_func):
    end = settings.config['end_date']
    m_filter = check_enter(end_date=end, strategy_fun=strategy_func)
    results = dict(filter(m_filter, stocks_data.items()))
    if len(results) > 0:
        push.strategy('**************"{0}"**************\n{1}\n**************"{0}"**************\n'.format(strategy, list(results.keys())))


def check_enter(end_date=None, strategy_fun=enter.check_volume):
    def end_date_filter(stock_data):
        if end_date is not None:
            if end_date < stock_data[1].iloc[0].日期:  # 该股票在end_date时还未上市
                logging.debug("{}在{}时还未上市".format(stock_data[0], end_date))
                return False
        return strategy_fun(stock_data[0], stock_data[1], end_date=end_date)


    return end_date_filter


# 统计数据
def statistics(all_data, stocks):
    limitup = len(all_data.loc[(all_data['涨跌幅'] >= 9.5)])
    limitdown = len(all_data.loc[(all_data['涨跌幅'] <= -9.5)])

    up5 = len(all_data.loc[(all_data['涨跌幅'] >= 5)])
    down5 = len(all_data.loc[(all_data['涨跌幅'] <= -5)])

    msg = "涨停数：{}   跌停数：{}\n涨幅大于5%数：{}  跌幅大于5%数：{}".format(limitup, limitdown, up5, down5)
    push.statistics(msg)


def prepare(today=None, today_ymd=None):
    """
    准备数据更新和因子更新
    
    Args:
        today: 日期字符串，格式为 'YYYY-MM-DD'，如果为None则使用当前日期
        today_ymd: 日期字符串，格式为 'YYYYMMDD'，如果为None则使用当前日期
    """
    # 如果参数为空，则获取当前时间
    if today is None or today_ymd is None:
        current_time = datetime.now()
        if today is None:
            today = current_time.strftime('%Y-%m-%d')
        if today_ymd is None:
            today_ymd = current_time.strftime('%Y%m%d')
    
    logging.info("[workflow] Starting all data updates...")
    daily_data_fetcher.run_all_updates(today=today, today_ymd=today_ymd)
    logging.info("[workflow] All data updates completed.")
    
    # 更新因子数据
    logging.info("[workflow] Starting factor updates...")
    update_all_factors_daily(today)
    logging.info("[workflow] All factor updates completed.")
    push_result(today=today, today_ymd=today_ymd)


def push_result(today=None, today_ymd=None):
    if today is None or today_ymd is None:
        current_time = datetime.now()
        if today is None:
            today = current_time.strftime('%Y-%m-%d')
        if today_ymd is None:
            today_ymd = current_time.strftime('%Y%m%d')
    import factors
    factor = factors.Custom400()
    df = factor.read_factor_file()
    df = df.loc[df['date'] == today]
    df = df.sort_values(by='value', ascending=False)
    df = df.head(15)
    codes = df['code'].tolist()
    stocks = factor.read_stock_basic_data(codes,None,None)
    daily_basic = factor.read_daily_basic_data(codes,today,1)
    df_daily_basic = data_reader.get_daily_basic_data(data_reader.list_available_stocks('daily'),today,15)
    df_daily_basic['limit_status'] = df_daily_basic['limit_status'].fillna(0)
    df_sorted = df_daily_basic.sort_values(by=['stock_code', 'trade_date'], ascending=[True, False])
    streak_series = df_sorted.groupby('stock_code').apply(count_consecutive_limit_up)
    result_df = streak_series.reset_index(name='limit_up_count')
    final_df = result_df[result_df['limit_up_count'] > 0].sort_values(by=['limit_up_count'], ascending=[False])
    stocks_limit = factor.read_stock_basic_data(final_df['stock_code'].tolist(),None,None)
    merge = df.merge(stocks, right_on='stock_code', left_on='code', how='left')
    merge = merge.merge(daily_basic, right_on='stock_code', left_on='code', how='left')
    merge2 = stocks_limit.merge(final_df,on='stock_code',how='left')
    merge2 = merge2[merge2['list_status']=='L']
    merge2['stock_label'] = merge2['stock_code'] + ' ' + merge2['name'] + '[' + merge2['industry'] + ']'
    msg = "以下股票可能有反弹，可关注：\n"
    for index, row in merge.iterrows():
        if row['turnover_rate'] > 2:
            msg += f"{row['code']} {row['name']} {'（跌停）' if row['limit_status']== -1 else ('（涨停）' if row['limit_status']== 1 else '')} 现价：{row['close']:.2f} 得分：{row['value']:.2f}\n"
    msg+="-----今日连板梯队-----\n"
    series = merge2.groupby('limit_up_count')['stock_label'].apply(list).sort_index(ascending=False)
    for count, labels in series.items():
        cn_count = num_map.get(count, str(count))
        content = ', '.join(labels)
        msg+=f"{cn_count}连板：{content}" if count>1 else f"{cn_count}板：{content}"
        msg+="\n"

    push.push(msg, today=today)

def count_consecutive_limit_up(group):
    # 获取该股票最近这几天的 limit_status
    statuses = group['limit_status'].values
    
    streak = 0
    for status in statuses:
        if status == 1:
            streak += 1
        else:
            # 一旦遇到不是涨停的，立即停止计数（连板中断）
            break
            
    return streak

if __name__ == '__main__':
    push_result(today='2025-11-05')