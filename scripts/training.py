import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from data_loader import load_data

def main():
    args = get_main_args()
    # set_granularity()
    set_cuda_devices(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    #     seed_everything(args.seed)
    # data_module = DataModule(args)
    # data_module.setup()
    ckpt_path = verify_ckpt_path(args)

    if ckpt_path is not None:
        model = NNUnet.load_from_checkpoint(ckpt_path, strict=False, args=args)
    else:
        model = NNUnet(args)
        
    # callbacks = [RichProgressBar(), ModelSummary(max_depth=2)]

    # what does this do?
    # if args.benchmark:
    #     batch_size = args.batch_size if args.exec_mode == "train" else args.val_batch_size
    #     filnename = args.logname if args.logname is not None else "perf.json"
    #     callbacks.append(
    #         LoggingCallback(
    #             log_dir=args.results,
    #             filnename=filnename,
    #             global_batch_size=batch_size * args.gpus * args.nodes,
    #             mode=args.exec_mode,
    #             warmup=args.warmup,
    #             dim=args.dim,
    #         )
    #     )
    # elif args.exec_mode == "train":
    #     if args.save_ckpt:
    #         callbacks.append(
    #             ModelCheckpoint(
    #                 dirpath=f"{args.ckpt_store_dir}/checkpoints",
    #                 filename="{epoch}-{dice:.2f}",
    #                 monitor="dice",
    #                 mode="max",
    #                 save_last=True,
    #             )
    #         )

    dataloaders = load_data(args.data, args.batch_size) ## ! NEED TO DEFINE batch_size somewhere! (and num_epochs) add to args?

    # Define your loss function and optimizer
    # ? where do we define these as well? args?
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # Train your model
    if args.exec_mode == "train":
        for epoch in range(args.num_epochs):
            model.train()
            for inputs, labels in dataloaders['train']:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    # Evaluate your model
    if args.exec_mode == "evaluate":
        model.eval()
        for inputs, labels in dataloaders['val']:
            outputs = model(inputs)
            # Perform evaluation calculations

    # Make predictions with your model
    if args.exec_mode == "predict":
        model.eval()
        for batch in dataloaders['test']:
            # are there no masks in this loader? will this return an error?
            inputs = batch
            outputs = model(inputs)
            # Process the predictions

            # process predictions here or get this to return the predicted probabilites and then
            # use another script to process?

if __name__ == "__main__":
    main()