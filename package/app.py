from flask import Flask, request, send_from_directory
from halo import Halo

flask_loader = Halo(text='Flask Loading', spinner='bouncingBar', color='white').start()
try:
	app = Flask(__name__)
except:
	flask_loader.fail(text="Flask failed to load")
	quit()
flask_loader.succeed(text="Flask Loaded")


print('==== Paraphrase Engine ====')
from compute import summarize

@app.get("/<path:path>")
def static_html(path):
	return send_from_directory('html', path)

@app.get("/css/<path:path>")
def static_css(path):
	return send_from_directory('css', path)

@app.get("/js/<path:path>")
def static_js(path):
	return send_from_directory('js', path)

@app.post("/compute")
def compute():
	text = request.json['text']
	return summarize(text)



develop = False

if __name__ == '__main__':
	#TODO: use an enviroment var for this
	#if (develop == True):
	app.run()
	#else:
	#	print('Starting production server')
	#	from waitress import serve
	#	serve(app)
	#	print('Started production server')