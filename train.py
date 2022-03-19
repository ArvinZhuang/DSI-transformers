from data import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback
from trainer import IndexingTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm


class QueryEvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs['model'].eval()
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True, ).reshape(inputs['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.log({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@10": hit_at_10 / len(self.test_dataset)})


def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(label[:np.where(label == 1)[0].item()],
                          predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}


def main():
    model_name = "t5-large"
    L = 32  # only use the first 32 tokens of documents (including title)

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login()
    wandb.init(project="DSI", name='NQ-10k-t5-large')

    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='cache')
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='cache')

    train_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_multi_task_train.json',
                                         max_length=L,
                                         cache_dir='cache',
                                         tokenizer=tokenizer)
    eval_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_multi_task_train.json',
                                        max_length=L,
                                        cache_dir='cache',
                                        tokenizer=tokenizer)

    test_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_valid.json',
                                        max_length=L,
                                        cache_dir='cache',
                                        tokenizer=tokenizer)

    ################################################################
    # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    ################################################################

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.0005,
        warmup_steps=10000,
        # weight_decay=0.01,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy='steps',
        eval_steps=1000,
        max_steps=1000000,
        dataloader_drop_last=False,  # necessary
        report_to='wandb',
        logging_steps=50,
        save_strategy='no',
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=10,
        gradient_accumulation_steps=2
    )

    trainer = IndexingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[QueryEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, tokenizer)],
        restrict_decode_vocab=restrict_decode_vocab
    )
    trainer.train(
    )


if __name__ == "__main__":
    main()
