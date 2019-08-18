MODEL = g104-b6c96-s97778688-d23397744

all: web_model

saved_model/saved_model.pb:
	python save_graph.py -name-scope swa_model -model-variables-prefix ../models/$(MODEL)/saved_model/variables/variables -model-config-json ../models/$(MODEL)/model.config.json

web_model: saved_model/saved_model.pb
	tensorflowjs_converter \
	    --input_format=tf_saved_model \
	    --saved_model_tags=serve \
	    saved_model \
	    web_model

clean:
	rm -rf saved_model web_model