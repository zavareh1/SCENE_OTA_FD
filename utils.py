import json
import os

__all__ = ["ConfLoader", "directory_setter", "config_overwriter"]


class ConfLoader:
    """
    Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key']
        """

        def __getattr__(self, key):
            if key in self:
                return self[key]
            # Important: raise AttributeError so hasattr(...) works correctly
            raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self):
        with open(self.conf_name, "r") as conf:
            opt = json.load(
                conf, object_hook=lambda d: self.DictWithAttributeAccess(d)
            )
        return opt

    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)
        return opt


def directory_setter(path="./results", make_dir=False):
    """
    Make directory if not exists.
    """
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # make dir if not exist
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )


def config_overwriter(opt, args):
    """
    Overwrite loaded configuration by parsing arguments.
    Safe for both schemas:
      - flat:   train_setups.n_rounds / local_epochs / sample_ratio / device
      - nested: train_setups.scenario.n_rounds / ...
    Also safely initializes algo.params before writing DS-FL overrides.
    """
    # --- data_setups ---
    if getattr(args, "dataset_name", None) is not None:
        opt.data_setups.dataset_name = args.dataset_name
    if getattr(args, "batch_size", None) is not None:
        opt.data_setups.batch_size = args.batch_size
    if getattr(args, "n_clients", None) is not None:
        opt.data_setups.n_clients = args.n_clients
    if getattr(args, "partition_method", None) is not None:
        opt.data_setups.partition.method = args.partition_method
    if getattr(args, "partition_s", None) is not None:
        opt.data_setups.partition.shard_per_user = args.partition_s
    if getattr(args, "partition_alpha", None) is not None:
        opt.data_setups.partition.alpha = args.partition_alpha
    if getattr(args, "unlabeled_pool_size", None) is not None:
        opt.data_setups.unlabeled_pool_size = int(args.unlabeled_pool_size)

    # --- model ---
    if getattr(args, "model_name", None) is not None:
        opt.train_setups.model.name = args.model_name

    # --- optimizer / scheduler params ---
    if getattr(args, "lr", None) is not None:
        opt.train_setups.optimizer.params.lr = args.lr
    if getattr(args, "momentum", None) is not None:
        opt.train_setups.optimizer.params.momentum = args.momentum
    if getattr(args, "wd", None) is not None:
        opt.train_setups.optimizer.params.weight_decay = args.wd

    # --- algo selection ---
    if getattr(args, "algo_name", None) is not None:
        opt.train_setups.algo.name = args.algo_name

    # --- seed ---
    if getattr(args, "seed", None) is not None:
        opt.train_setups.seed = args.seed

    # --- wandb ---
    if getattr(args, "group", None) is not None:
        opt.wandb_setups.group = args.group
    if getattr(args, "exp_name", None) is not None:
        opt.wandb_setups.name = args.exp_name

    # ----- Rounds / epochs / ratio / device (schema-agnostic) -----
    if getattr(args, "n_rounds", None) is not None:
        if "scenario" in opt.train_setups:
            opt.train_setups.scenario.n_rounds = int(args.n_rounds)
        else:
            opt.train_setups.n_rounds = int(args.n_rounds)

    if getattr(args, "local_epochs", None) is not None:
        if "scenario" in opt.train_setups:
            opt.train_setups.scenario.local_epochs = int(args.local_epochs)
        else:
            opt.train_setups.local_epochs = int(args.local_epochs)

    if getattr(args, "sample_ratio", None) is not None:
        if "scenario" in opt.train_setups:
            opt.train_setups.scenario.sample_ratio = float(args.sample_ratio)
        else:
            opt.train_setups.sample_ratio = float(args.sample_ratio)

    if getattr(args, "device", None) is not None:
        if "scenario" in opt.train_setups:
            opt.train_setups.scenario.device = args.device
        else:
            opt.train_setups.device = args.device

    if getattr(args, "eval_interval", None) is not None:
        if "scenario" in opt.train_setups:
            opt.train_setups.scenario.eval_interval = int(args.eval_interval)
        else:
            opt.train_setups.eval_interval = int(args.eval_interval)

    # ----- Ensure algo.params exists before writing DS-FL overrides -----
    if ("params" not in opt.train_setups.algo) or (opt.train_setups.algo.params is None):
        opt.train_setups.algo.params = ConfLoader.DictWithAttributeAccess({})

    if getattr(args, "unlabeled_per_round", None) is not None:
        opt.train_setups.algo.params["unlabeled_per_round"] = int(args.unlabeled_per_round)

    if getattr(args, "unlabeled_batch_size", None) is not None:
        opt.train_setups.algo.params["unlabeled_batch_size"] = int(args.unlabeled_batch_size)

    if getattr(args, "client_distill_epochs", None) is not None:
        opt.train_setups.algo.params["client_distill_epochs"] = int(args.client_distill_epochs)

    if getattr(args, "server_distill_epochs", None) is not None:
        opt.train_setups.algo.params["server_distill_epochs"] = int(args.server_distill_epochs)

    return opt
