# -*- encoding: UTF-8 -*-

import logging
import settings
import smtplib
from email.mime.text import MIMEText
import utils


def push(body):
    if settings.config['push']['enable']:
        msg = MIMEText(body)
        msg['Subject'] = f'股票策略提醒 {utils.get_today_str()}'
        msg['From'] = settings.config['push']['smtp']['from_email']
        msg['To'] = settings.config['push']['smtp']['to_email']

        try:
            server = smtplib.SMTP_SSL(
                settings.config['push']['smtp']['server'],
                settings.config['push']['smtp']['port']
            )
            server.login(
                settings.config['push']['smtp']['username'],
                settings.config['push']['smtp']['password']
            )
            server.send_message(msg)
            server.quit()
            logging.info("邮件发送成功")
        except Exception as e:
            logging.error(f"邮件发送失败: {str(e)}")
    logging.info(body)


def statistics(msg=None):
    push(msg)


def strategy(msg=None):
    if msg is None or not msg:
        msg = '今日没有符合条件的股票'
    push(msg)
