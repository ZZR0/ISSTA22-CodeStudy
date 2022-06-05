export PYTHONPATH='../../Vuld_SySe/':$PYTHONPATH
for i in 1 2 3 4 5; do
	python ../../Vuld_SySe/vul_det_main.py \
		--train_file ../../data/SySeVR/Array_usage-chrome_debian.json\
		--model_path ../../models/sys_array_train_on_chrome_debian.bin \
		--word_to_vec ../../data/Word2Vec/li_et_al_wv \
		--batch_size 32 \
		--model_type bigru >> ../../outputs/sys_array_train_on_chrome_debian.log
done
