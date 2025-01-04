# %%
import os
import torch
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from HG import HG
from dataset import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error

# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        drug, pock, comp, esm_fea = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        with torch.no_grad():
            pred = model(drug, pock, comp, esm_fea)
            label = data[0].y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig'
    config = Config(cfg)
    args = config.get_config()
    gpu = args.get("gpu")
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    num_works = args.get("num_works")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")

    data_path = os.path.join(data_root, 'train')
    test2013_dir = os.path.join(data_root, 'test_2013')
    test2016_dir = os.path.join(data_root, 'test_2016')
    test_hiq_dir = os.path.join(data_root, 'test_hiq')

    train_df = pd.read_csv(os.path.join(data_root, "train.csv")).sample(frac=1., random_state=123)
    valid_df = pd.read_csv(os.path.join(data_root, "valid.csv")).sample(frac=1., random_state=123)
    test_df = pd.read_csv(os.path.join(data_root, 'test.csv'))
    test2013_df = pd.read_csv(os.path.join(data_root, 'test_2013.csv'))
    test2016_df = pd.read_csv(os.path.join(data_root, 'test_2016.csv'))
    test_hiq_df = pd.read_csv(os.path.join(data_root, 'test_hiq.csv'))

    train_set = GraphDataset(data_path, train_df, graph_type=graph_type, create=False)
    valid_set = GraphDataset(data_path, valid_df, graph_type=graph_type, create=False)
    test_set = GraphDataset(data_path, test_df, graph_type=graph_type, create=False)
    test_2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
    test_2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
    test_hiq_set = GraphDataset(test_hiq_dir, test_hiq_df, graph_type=graph_type, create=False)

    train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_works)
    valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_works)
    test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_works)
    test_2013_loader = PLIDataLoader(test_2013_set, batch_size=batch_size, shuffle=False, num_workers=num_works)
    test_2016_loader = PLIDataLoader(test_2016_set, batch_size=batch_size, shuffle=False, num_workers=num_works)
    test_hiq_loader = PLIDataLoader(test_hiq_set, batch_size=batch_size, shuffle=False, num_workers=num_works)

    for repeat in range(repeats):
        args['repeat'] = repeat
        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)

        logger.info(f"train data: {len(train_set)}\n valid data: {len(valid_set)}\n test data: {len(test_set)}\n "
                    f"test_2013 data: {len(test_2013_set)}\n test_2016 data: {len(test_2016_set)}\n test_hiq data: {len(test_hiq_set)}")

        device = torch.device(f'cuda:{gpu}')
        model = HG(35, 256, 1, n_layers=args.get("n_layers"), normalize=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []

        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                drug, pock, comp, esm_fea = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                pred = model(drug, pock, comp, esm_fea)
                label = data[0].y

                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)
            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
                    # log test
                    test_rmse, test_pr = val(model, test_loader, device)
                    test_2013_rmse, test_2013_pr = val(model, test_2013_loader, device)
                    test_2016_rmse, test_2016_pr = val(model, test_2016_loader, device)
                    test_hiq_rmse, test_hiq_pr = val(model, test_hiq_loader, device)
                    best_msg = "test_rmse-%.4f, test_pr-%.4f, test_2013_rmse-%.4f, test_2013_pr-%.4f, test_2016_rmse-%.4f, test_2016_pr-%.4f, test_hiq_rmse-%.4f, test_hiq_pr-%.4f" \
                               % (test_rmse, test_pr, test_2013_rmse, test_2013_pr, test_2016_rmse, test_2016_pr,
                                  test_hiq_rmse, test_hiq_pr)
                    logger.info(f'epoch-{epoch} best test:\n{best_msg}')
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break
# %%