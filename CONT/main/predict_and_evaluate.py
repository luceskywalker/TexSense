import os
from CONT.utils.data import PiImuToMomentsDataset as DATA_SET
from CONT.utils.model import PiImutoMomentsNet as MODEL_CLASS
from CONT.utils.utils import *
from tqdm import tqdm
from CONT.utils.data import create_dataloader
import matplotlib.pyplot as plt
from CONT.utils.plot import plot_prediction as PLOT
from CONT.utils.utils import model_statistics
from CONT.utils.statistics import get_model_performance

# go to root directory
os.chdir("..")

# get device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

for study in ["study1", "study2"]:
	models = os.listdir(f"models/{study}")

	for model in tqdm(models):
		print(model[:-3])
		# load model_dict
		model_dict = torch.load(f"models/{study}/{model}", weights_only=False, map_location=device)

		# get state dict, config, model class and init
		STATE_DICT = model_dict["STATE_DICT"]
		cfg = model_dict["cfg"]
		MODEL = MODEL_CLASS(cfg)
		MODEL.load_state_dict(STATE_DICT)

		# create val-loader
		_, val_loader = create_dataloader(cfg, DATA_SET, val_only=True)

		# predict and get performance metrics
		cont, phss = get_model_performance(cfg, MODEL, device, val_loader)

		# clean up and get output dfs
		cont_df = dict_to_multilevel_df(cont)
		phss_df = dict_to_multilevel_df(phss)

		# save results
		cont_df.to_csv(f"performance/{study}/{model}_cont.csv")
		phss_df.to_csv(f"performance/{study}/{model}_phss.csv")

