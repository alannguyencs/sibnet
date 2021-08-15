from sys import stdout
import logging

class Allog():
	def __init__(self, log_file_path):
		logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

		self.log = logging.getLogger()
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler = logging.StreamHandler(stdout)
		handler.setFormatter(formatter)
		self.log.addHandler(handler)

	def info(self, message):
		self.log.info(message)