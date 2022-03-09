from TDA_KE.data import get_KP_dataloader, truncate_sequence
from flair.models import SequenceTagger
from flair.embeddings import StackedEmbeddings, TransformerWordEmbeddings
from transformers import BertTokenizer
from flair.trainers import ModelTrainer
import torch, flair


if __name__ == "__main__":
    flair.device = torch.device("cuda:1")
    corpus = get_KP_dataloader(dataset="SE-2017", dev=True, test=True)
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    max_seq_len, tag_type = 512, "BIO"
    embeddings = StackedEmbeddings(
        embeddings=[TransformerWordEmbeddings("bert-large-cased")]
    )
    label_dict = corpus.make_label_dictionary(label_type=tag_type)
    tagger = SequenceTagger(
        hidden_size=128,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type=tag_type,
        use_crf=False,
        rnn_layers=1,
        dropout=0,
        word_dropout=0.05,
        locked_dropout=0.5,
    )
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(
        "./logs",
        learning_rate=0.05,
        mini_batch_size=12,
        anneal_factor=0.5,
        patience=4,
        max_epochs=10,
        param_selection_mode=False,
        num_workers=24,
    )
