class SimpleLoadingBar:
	def __init__(self):
		self.prog = 0#Always writen in decimal
		self.msg = ""
		self.maxout = 0
	
	#Updates progress and message. If message is not defined then the message will be removed.
	def Update(self, prog, **kwar):
		self.msg = ""
		if "msg" in kwar:
			self.msg =  " : "+kwar["msg"]
			msg =  " : "+kwar["msg"]
		self.prog = prog
		out = "\r[{0}{1}]{2}%{3}".format("#"*int(20*self.prog), " "*(20-int(20*self.prog)), self.prog*100, self.msg)
		if self.maxout > len(out):
			out += " "*(self.maxout-len(out))
		else:
			self.maxout = len(out)
		print(out, end="")
		
	#Makes one step. Does not update message if message is not defined
	def Step(self, stepsize = 0.01, **kwar):
		if "msg" in kwar:
			self.msg = kwar["msg"]
		self.Update(self.prog + stepsize)
		
	#Start a new loadingbar
	def Start(self, **kwar):
		self.prog = 0
		self.maxout = 0
		self.msg = ""
		if "msg" in kwar:
			self.msg = " : "+kwar["msg"]
		self.Update(self.prog)
		
	#Sets progress to 100% and updates message. If message is not defined then the message will be removed.
	def Finnish(self, **kwar):
		self.msg = ""
		self.prog = 1
		if "msg" in kwar:
			self.msg = " : "+kwar["msg"]
		out = "\r[{0}]{1}%{2}".format("#"*20, self.prog*100, self.msg)
		if self.maxout > len(out):
			out += " "*(self.maxout-len(out))
		else:
			self.maxout = len(out)
		print(out)

#Some testing
if __name__ == "__main__":
	import time
	e = SimpleLoadingBar()
	e.Start(msg = "test")
	time.sleep(1)
	e.Step()
	time.sleep(1)
	e.Step()
	time.sleep(1)
	e.Step(0.005)
	time.sleep(1)
	e.Update(0.5, msg = "more testing")
	time.sleep(1)
	e.Finnish(msg = "Done")
	print("Done")
	e.Start()
	time.sleep(1)
	e.Finnish()
