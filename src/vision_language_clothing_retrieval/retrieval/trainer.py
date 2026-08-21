import torch
from torch.utils.data import DataLoader

from vision_language_clothing_retrieval.retrieval.loss import (
    contrastive_loss,
)
from vision_language_clothing_retrieval.retrieval.model import (
    MultimodalModel,
)


class MultimodalTrainer:
    """Trenira projekcione slojeve pomoću contrastive loss-a."""

    def __init__(
        self,
        model: MultimodalModel,
        learning_rate: float = 1e-3,
        temperature: float = 0.07,
    ) -> None:
        self.model = model
        self.temperature = temperature

        # Adam optimizator ažurira parametre projekcionih slojeva
        # na osnovu gradijenata dobijenih tokom backpropagation-a.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        device: str = "cpu",
    ) -> float:
        # Model se postavlja u training režim kako bi se omogućilo
        # učenje njegovih parametara.
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Dataset se obrađuje batch po batch.
        for batch in dataloader:
            # Uzimaju se već generisani image i text embeddingi
            # i prebacuju na izabrani uređaj.
            image_embeddings = batch["image_embeddings"].to(device)
            text_embeddings = batch["text_embeddings"].to(device)

            # Brišu se gradijenti iz prethodnog koraka optimizacije.
            self.optimizer.zero_grad()

            # Originalni image i text embeddingi prolaze kroz
            # odgovarajuće projekcione slojeve u zajednički prostor.
            projected_images, projected_texts = self.model(
                image_embeddings,
                text_embeddings,
            )

            # Contrastive loss poredi image i text reprezentacije
            # i podstiče odgovarajuće parove da budu bliži.
            loss = contrastive_loss(
                projected_images,
                projected_texts,
                temperature=self.temperature,
            )

            # Računaju se gradijenti loss-a u odnosu na parametre modela.
            loss.backward()

            # Parametri projekcionih slojeva se ažuriraju pomoću
            # izračunatih gradijenata.
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Vraća se prosečan loss preko svih obrađenih batch-eva.
        return total_loss / num_batches
