import logging; LOGGER = logging.getLogger(__name__)
from torch.utils.data import DataLoader
import torch

from devices import DEVICE
import models
import datasets

class SimpleAdversarialTrainer:
    def __init__(self,epochs=25,datset_size:int=1024) -> None:

        self.EPOCHS = epochs
        self.reconstruction_loss_mult = 1.0
        self.adversarial_loss_mult = 0.002

        self.load_datset(
            num_images=datset_size,  # Images to load each epoch, maybe breaks if bigger then num images in datset --\_( 0 _ 0 )_/-- ( too lazy to explain )
            batch_size=128    # How many images to run trough the model at once during training
        )
        self.load_models()
        self.create_optimizers_and_critereons()
    
    def create_optimizers_and_critereons(self) -> None:
        self.optimizer_auto_encoder  = torch.optim.Adam(self.model.model_auto_encoder.parameters(), lr=0.0002125)
        self.optimizer_discriminator = torch.optim.Adam(self.model.model_discriminator.parameters(), lr=0.0002125)

        self.reconstruction_criterion = torch.nn.MSELoss()
        self.adversarial_criterion    = torch.nn.BCELoss()
    
    def train(self):

        for epoch in range(self.EPOCHS):
            self.train_auto_encoder_epoch(epoch)
            self.train_discriminator_epoch(epoch)
        
        self.model.save()
    
    def train_auto_encoder_epoch(self,epoch) -> None:

        # Train the autoencoder
        self.freezer(fase_is_encoder=True )

        # Run code for each batch
        running_loss = 0.0
        for batch in self.auto_encoder_dataloader:

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
        avg_loss = running_loss / len(self.auto_encoder_dataloader)
        LOGGER.info(f"AutoEncoder - Epoch {epoch+1}/{self.EPOCHS} avg. loss: {avg_loss:.4f}")
    
    def train_discriminator_epoch(self,epoch) -> None:

        # Train the autoencoder
        self.freezer(fase_is_encoder=False)

        # Run code for each batch
        running_loss = 0.0
        for batch in self.discriminator_dataloader:

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

            total_loss = loss_real + loss_fake

            #
            # backward pass
            #

            self.optimizer_discriminator.zero_grad()
            total_loss.backward()
            self.optimizer_discriminator.step()

            running_loss += total_loss.item()
        
        # Calculate epoch loss
        avg_loss = running_loss / len(self.discriminator_dataloader)
        LOGGER.info(f"Discriminator - Epoch {epoch+1}/{self.EPOCHS} avg. loss: {avg_loss:.4f}")
    
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
    
    def transform_batch(self,batch):
        """
        Takes in a batch from the datset and converts it into something that the neural nets understand
        
        Args:
            batch: batch to transform

        Returns:
            Transformed batch
        """        
        return batch.view(-1, 1, 64, 64).to(DEVICE)

    def load_models(self) -> None:
        """
        Create the neural nets
        """        
        self.model = models.SimpleAdversarialNetwork()
    
    def load_datset(self,num_images:int=4096,batch_size:int=128) -> None:
        """
        Loads the datset
        """
        # Load half of the images for the auto encoder and the other half for the discriminator
        auto_encoder_dataset  = datasets.TorchDataset("ashwingupta3012_human_faces",limit=int(num_images*0.5),starting_index=0)
        discriminator_dataset = datasets.TorchDataset("ashwingupta3012_human_faces",limit=int(num_images*0.5),starting_index=int(num_images*0.5))

        self.auto_encoder_dataloader  = DataLoader(dataset=auto_encoder_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
        self.discriminator_dataloader = DataLoader(dataset=discriminator_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    
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
