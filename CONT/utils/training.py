import torch


class TrainingRoutine:
    """Class for training routine of a neural network model."""

    def __init__(
            self,
            cfg,
            device,
            model,
            criterion,
            train_loader,
            val_loader
    ):
        """
        Initialize the training routine.

        Args:
            cfg (ConfigParser): ConfigParser object containing experiment parameters.
            device (torch.device): Device on which to perform training.
            model (torch.nn.Module): Neural network model to be trained.
            criterion (torch.nn.Module): Loss function criterion.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg.getfloat('training_parameters', 'min_learning_rate'))
        self.criterion = criterion
        self.criterion_str = str(criterion).split("(")[0]
        self.train_loader = train_loader
        self.train_losses = []
        self.val_loader = val_loader
        self.val_losses = []
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                         max_lr=self.cfg.getfloat('training_parameters', 'max_learning_rate'),  # Upper learning rate boundaries in the cycle for each parameter group
                         steps_per_epoch=5,  # The number of steps per epoch to train for.
                         epochs=self.cfg.getint('experiment', 'n_epochs') // 5,  # The number of epochs to train for.
                         anneal_strategy='linear')  # Specifies the annealing strategy
        if cfg.getboolean("experiment", "extra_noise"):
            self.noise_ratio = cfg.getfloat("model_parameters", "noise_ratio")

    def add_noise(self, sample):
        """
        Add noise to the input sample.

        Args:
            sample (torch.Tensor): Input sample.

        Returns:
            torch.Tensor: Noisy sample.
        """
        # Perform the cumulative sum to simulate the walk
        rnd = torch.randn_like(sample) * self.noise_ratio
        # Perform the cumulative sum to simulate the walk
        random_walk = rnd.cumsum(dim=0)
        # Create a tensor that represents the decay factor for each step
        decay_factors = torch.arange(1, sample.shape[0] + 1).view(-1, 1).to(self.device)
        # Apply the decay to the random walk
        random_walk = random_walk / decay_factors
        return (sample + random_walk).view_as(sample)

    def train(self, epoch, verbose=0):
        """
        Perform training for one epoch.

        Args:
            epoch (int): Current epoch number.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        run_loss = 0.0

        self.model.train()
        for _, sample in enumerate(self.train_loader):
            pi_data = sample["pi"].to(self.device)
            moments = sample["moments"].to(self.device)
            imu = sample["imu"].to(self.device)

            if self.cfg.getboolean("experiment", "extra_noise"):
                pi_data = torch.stack([self.add_noise(samp) for samp in pi_data])
                imu = torch.stack([self.add_noise(samp) for samp in imu])

            self.optimizer.zero_grad()
            aux = sample['level'].to(self.device)
            pred = self.model(pi=pi_data, aux=aux, imu=imu)
            loss = self.criterion(pred, moments)
            loss.backward()
            self.optimizer.step()
            run_loss += loss.item()
        train_loss = run_loss / len(self.train_loader)
        self.train_losses.append(train_loss)

        # Step the LR scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        if verbose >= 1:
            print(f"Training loss {train_loss}")

    def val(self, epoch, verbose=0):
        """
        Perform validation for one epoch.

        Args:
            epoch (int): Current epoch number.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        run_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for sample in self.val_loader:
                pi_data = sample["pi"].to(self.device)
                moments = sample["moments"].to(self.device)
                imu = sample["imu"].to(self.device)
                self.optimizer.zero_grad()

                aux = sample['level'].to(self.device)

                pred = self.model(pi=pi_data, aux=aux, imu=imu)
                loss = self.criterion(pred, moments)
                run_loss += loss.item()

            val_loss = run_loss / len(self.val_loader)
            self.val_losses.append(val_loss)
            if verbose >= 1:
                print(f"Validation loss {val_loss}")
