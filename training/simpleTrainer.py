import logging; LOGGER = logging.getLogger(__name__)
import torch

from devices import DEVICE
import models

from training import ImageTrainertemplate

class SimpleAdversarialTrainer(ImageTrainertemplate):
    def train_epoch(self,epoch:int) -> None:
        self.train_auto_encoder_epoch(epoch)
        self.train_discriminator_epoch(epoch)

        self.epoch_bar.set_postfix({
            "Loss: ae:":f"{self.auto_encoder_avg_loss:.4f}",
            "di:":f"{self.discriminator_avg_loss:.4f}"
        })

    def create_optimizers_and_critereons(self) -> None:
        self.reconstruction_loss_mult = 1.0
        self.adversarial_loss_mult = 0.05

        self.optimizer_auto_encoder  = torch.optim.Adam(self.model.model_auto_encoder.parameters(),  lr=self.learn_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.model.model_discriminator.parameters(), lr=self.learn_rate)

        self.reconstruction_criterion = torch.nn.MSELoss()
        self.adversarial_criterion    = torch.nn.BCEWithLogitsLoss()
    
    def train_auto_encoder_epoch(self,epoch) -> None:

        # Train the autoencoder
        self.freezer(fase_is_encoder=True )

        # Run code for each batch
        running_loss = 0.0
        for batch in self.data_loader:

            # Forward pass
            transformed_batch = self.transform_batch(batch)
            auto_encoded_batch = self.model.model_auto_encoder(transformed_batch)
            loss = self.calculate_auto_encoder_loss( transformed_batch, auto_encoded_batch )

            # Backward pass
            self.model.model_auto_encoder.zero_grad()
            loss.backward()
            self.optimizer_auto_encoder.step()

            # Add up loss
            running_loss += loss.item()
        
        # Calculate epoch loss
        self.auto_encoder_avg_loss = running_loss / len(self.data_loader)
    
    def train_discriminator_epoch(self,epoch) -> None:

        # Train the autoencoder
        self.freezer(fase_is_encoder=False)

        # Run code for each batch
        running_loss = 0.0
        for batch in self.data_loader:

            transformed_batch = self.transform_batch(batch)

            #
            # Forward pass
            #

            # Train on real images
            predicted = self.model.model_discriminator(transformed_batch)
            correct_prediction = torch.ones_like(predicted).to(DEVICE) # 1 = real
            loss_real = self.adversarial_criterion(predicted, correct_prediction)

            # Train on genrated images
            fake_images = self.model.model_auto_encoder(transformed_batch).detach() # Create a set of generated images
            predicted = self.model.model_discriminator(fake_images)
            correct_prediction = torch.zeros_like(predicted).to(DEVICE) # 0 = fake
            loss_fake = self.adversarial_criterion(predicted, correct_prediction)

            total_loss = (loss_real + loss_fake)

            #
            # backward pass
            #

            self.optimizer_discriminator.zero_grad()
            total_loss.backward()
            self.optimizer_discriminator.step()

            running_loss += total_loss.item()
        
        # Calculate epoch loss
        self.discriminator_avg_loss = running_loss / len(self.data_loader)
    
    def calculate_auto_encoder_loss(self,original_batch,auto_encoded_batch):

        # calculate pixel-pixel comaprison
        reconstruction_loss = self.reconstruction_criterion(auto_encoded_batch,original_batch)

        # Calculate 'realism' suing the discriminator
        discriminator_predictions = self.model.model_discriminator(auto_encoded_batch)
        real_labels = torch.ones_like(discriminator_predictions).to(DEVICE) # this is what we try to achieve all ones ( means all realism )
        adversarial_loss = self.adversarial_criterion(discriminator_predictions, real_labels) # see how good we achieved the realism

        # now combine them
        loss = ( reconstruction_loss * self.reconstruction_loss_mult ) + ( adversarial_loss * self.adversarial_loss_mult )

        return loss
 
    def load_models(self) -> None:
        """
        Create the neural nets
        """        
        self.model = models.SimpleAdversarialNetwork()
    
    def freezer(self, fase_is_encoder: bool) -> None:
        """
        Freezes and unfreezes paemters of the auto encoder / discirminattor respecitvly

        Args:
            fase_is_encoder (bool): If encoder then it will freeze the discirmntor and ...
        """        
        if fase_is_encoder:
            # Freeze discriminator / Unfreeze autoencoder
            self.model.model_discriminator.eval()
            for param in self.model.model_discriminator.parameters():
                param.requires_grad = False
                
            self.model.model_auto_encoder.train(True)
            for param in self.model.model_auto_encoder.parameters():
                param.requires_grad = True
        else:
            # Freeze autoencoder / Unfreeze discriminator
            self.model.model_auto_encoder.eval()
            for param in self.model.model_auto_encoder.parameters():
                param.requires_grad = False
                
            self.model.model_discriminator.train(True)
            for param in self.model.model_discriminator.parameters():
                param.requires_grad = True