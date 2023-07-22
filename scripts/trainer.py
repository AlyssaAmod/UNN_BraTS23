from monai_functions import *
from utils.utils import get_main_args

"""General Setup: 
    logging,
    utils.args 
    seed,
    cuda, 
    root dir"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
args = get_main_args()
set_determinism(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# root_dir = args.data
# results_dir = args.results

model, n_channels = define_model(args.ckpt_path)

# Print out model architecture
# print(model)

dataloaders = define_dataloaders(n_channels)
train_loader, val_loader = dataloaders['train'], dataloaders['val']

optimiser, criterion, lr_scheduler = model_params(args, model)

# TRAIN MODEL
train(args, model, device, train_loader, val_loader, optimiser, criterion, lr_scheduler)