# -*- encoding: UTF-8 -*-

import logging
import settings
import smtplib
from email.mime.text import MIMEText


def push(body):
    if settings.config['push']['enable']:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = "your_email@example.com"
        msg['To'] = to_email

        server = smtplib.SMTP_SSL('smtp.example.com', 465)
        server.login("your_email@example.com", "your_password")
        server.send_message(msg)
        server.quit()
        print(body)
    logging.info(body)


def statistics(msg=None):
    push(msg)


def strategy(msg=None):
    if msg is None or not msg:
        msg = '今日没有符合条件的股票'
    push(msg)
