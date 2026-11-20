import os
import sys
sys.path.append('../')
sys.path.append('../../')
import logging
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as FUNS
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from optimization import AdamW, Warmup
from data_utils import SATokenizer
import data_utils
import modelconfig
from run_base_sa import BertForClassification

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_orthogonal_loss(feat1, feat2):
    feat1_norm = FUNS.normalize(feat1, p=2.0, dim=1)
    feat2_norm = FUNS.normalize(feat2, p=2.0, dim=1)
    # Element-wise dot product per sample (cosine similarity squared)
    cos_sim = torch.sum(feat1_norm * feat2_norm, dim=1)
    return torch.mean(torch.square(cos_sim))


class TokenAttention(nn.Module):
    """Learns token-level importance weights over BERT's sequence output."""
    def __init__(self, hidden_size):
        super(TokenAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, sequence_output, attention_mask):
        # sequence_output: [B, L, H], attention_mask: [B, L]
        scores = self.attn(sequence_output).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)           # [B, L]
        context = torch.bmm(weights.unsqueeze(1), sequence_output).squeeze(1)  # [B, H]
        return context, weights


class IntermediateMLP(nn.Module):
    def __init__(self, config):
        super(IntermediateMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU()
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)


class BertForDomain(BertPreTrainedModel):
    all_tied_weights_keys = {}
    _tied_weights_keys = []

    def __init__(self, config):
        super(BertForDomain, self).__init__(config)
        config.return_dict = False
        self.bert = BertModel(config)
        self.domain_attn = TokenAttention(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.domain_classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, return_weights=False):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]
        pooled_output, domain_weights = self.domain_attn(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)
        if return_weights:
            return pooled_output, domain_weights
        return pooled_output


class BertaForClassificationModel(BertPreTrainedModel):
    all_tied_weights_keys = {}
    _tied_weights_keys = []

    def __init__(self, config, num_labels=2, *model_args, **model_kwargs):
        super(BertaForClassificationModel, self).__init__(config)
        config.return_dict = False
        self.bert = BertModel(config)
        self.config = config
        self.num_labels = num_labels

        self.sent_attn = TokenAttention(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_feature = IntermediateMLP(config)
        self.sent_classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.domain = BertForDomain(config)

        self.KL_loss = nn.KLDivLoss(reduction="batchmean")
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.softmax = nn.Softmax(dim=-1)

    def confidence_gated_contrastive_loss(self, instances, proto, pseudo_labels, tao=0.05, threshold=0.85):
        max_probs, _ = torch.max(pseudo_labels, dim=1)
        mask = max_probs >= threshold
        if mask.sum() == 0:
            return torch.tensor(0.0, device=instances.device)

        filtered_instances = instances[mask]
        filtered_targets = torch.argmax(pseudo_labels[mask], dim=1)
        norm_features = FUNS.normalize(filtered_instances, p=2.0, dim=1)
        norm_proto = FUNS.normalize(proto, p=2.0, dim=-1)
        cosim = torch.matmul(norm_features, norm_proto.t())
        return self.CE_loss(cosim / 0.1, filtered_targets)

    def forward(self, src_input_ids=None, src_attention_mask=None, src_labels=None,
                tgt_input_ids=None, tgt_attention_mask=None, pseudo_labels=None, return_weights=False):

        # Stage A: Train Domain Discriminator
        if src_input_ids is not None and tgt_input_ids is not None and src_labels is None:
            src_batch_size, _ = src_input_ids.size()
            tgt_batch_size, _ = tgt_input_ids.size()
            src_domain_feature = self.domain(input_ids=src_input_ids, attention_mask=src_attention_mask)
            tgt_domain_feature = self.domain(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask)

            src_domain_loss = self.CE_loss(
                self.domain.domain_classifier(src_domain_feature),
                torch.zeros(src_batch_size, dtype=torch.long, device=src_input_ids.device)
            )
            tgt_domain_loss = self.CE_loss(
                self.domain.domain_classifier(tgt_domain_feature),
                torch.ones(tgt_batch_size, dtype=torch.long, device=src_input_ids.device)
            )
            zero = torch.tensor(0.0, device=src_input_ids.device)
            return zero, src_domain_loss + tgt_domain_loss, zero, zero

        # Stage B: Train Disentangled Sentiment + Attention + Prototypical Loss
        if src_labels is not None and pseudo_labels is not None:
            src_batch_size, _ = src_input_ids.size()
            tgt_batch_size, _ = tgt_input_ids.size()

            src_outputs = self.bert(src_input_ids, attention_mask=src_attention_mask, return_dict=False)
            src_pooled, _ = self.sent_attn(src_outputs[0], src_attention_mask)
            src_sent_feature = self.sentiment_feature(self.dropout(src_pooled))
            src_sent_loss = self.CE_loss(self.sent_classifier(src_sent_feature), src_labels)

            tgt_outputs = self.bert(tgt_input_ids, attention_mask=tgt_attention_mask, return_dict=False)
            tgt_pooled, _ = self.sent_attn(tgt_outputs[0], tgt_attention_mask)
            tgt_sent_feature = self.sentiment_feature(self.dropout(tgt_pooled))

            # Disentanglement: KL divergence + Active Orthogonal Loss
            src_op_domain_loss = self.KL_loss(
                FUNS.log_softmax(self.domain.domain_classifier(src_sent_feature), dim=1),
                torch.full((src_batch_size, 2), fill_value=0.5, dtype=torch.float, device=src_input_ids.device)
            )
            tgt_op_domain_loss = self.KL_loss(
                FUNS.log_softmax(self.domain.domain_classifier(tgt_sent_feature), dim=1),
                torch.full((tgt_batch_size, 2), fill_value=0.5, dtype=torch.float, device=src_input_ids.device)
            )

            with torch.no_grad():
                src_dom_feat = self.domain(input_ids=src_input_ids, attention_mask=src_attention_mask)
                tgt_dom_feat = self.domain(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask)
            ortho_loss = get_orthogonal_loss(src_sent_feature, src_dom_feat) + get_orthogonal_loss(tgt_sent_feature, tgt_dom_feat)

            disentangle_loss = 0.1 * (src_op_domain_loss + tgt_op_domain_loss) + 0.1 * ortho_loss

            # Confidence-gated prototypical alignment
            tgt_sim_loss = self.confidence_gated_contrastive_loss(
                instances=tgt_sent_feature,
                proto=self.sent_classifier.weight,
                pseudo_labels=pseudo_labels,
                tao=0.05,
                threshold=0.85
            )
            return src_sent_loss, torch.tensor(0.0, device=src_input_ids.device), disentangle_loss, tgt_sim_loss

        # Inference / Evaluation
        else:
            outputs = self.bert(src_input_ids, attention_mask=src_attention_mask, return_dict=False)
            src_pooled, sent_weights = self.sent_attn(outputs[0], src_attention_mask)
            src_sent_feature = self.sentiment_feature(self.dropout(src_pooled))
            src_domain_feature, domain_weights = self.domain(input_ids=src_input_ids, attention_mask=src_attention_mask, return_weights=True)
            logits = self.sent_classifier(src_sent_feature)
            if return_weights:
                return logits, sent_weights, domain_weights
            return src_pooled, src_sent_feature, src_domain_feature, logits


def train(args, model, train_dataset, pseudo_labeled_dataset, eval_dataset):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    pseudo_labeled_dataloader = DataLoader(pseudo_labeled_dataset, sampler=RandomSampler(pseudo_labeled_dataset), batch_size=args.train_batch_size)
    num_train_steps = int(len(train_dataloader)) * args.num_train_epochs

    def creatOptimizer(named_parameters):
        param_optimizer = [(k, v) for k, v in named_parameters if v.requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.warmup_proportion)
        scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_train_steps)
        return optimizer, scheduler

    def steps(loss, optimizer, scheduler):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    model.to(args.device)
    sent_params = (
        list(model.bert.named_parameters()) +
        list(model.sent_attn.named_parameters()) +
        list(model.sentiment_feature.named_parameters()) +
        list(model.sent_classifier.named_parameters())
    )
    sent_optimizer, sent_scheduler = creatOptimizer(sent_params)
    domain_optimizer, domain_scheduler = creatOptimizer(model.domain.named_parameters())

    logger.info("Batch size = %d | Total steps = %d", args.train_batch_size, num_train_steps)
    model.zero_grad()
    model.train()
    global_step = 0

    for epoch in range(int(args.num_train_epochs)):
        target_dataloader = iter(pseudo_labeled_dataloader)
        for step, batch in enumerate(train_dataloader):
            src_input_ids, src_input_mask, src_label_ids = tuple(t.to(args.device) for t in batch)
            try:
                tgt_batch = next(target_dataloader)
            except StopIteration:
                target_dataloader = iter(pseudo_labeled_dataloader)
                tgt_batch = next(target_dataloader)
            tgt_input_ids, tgt_input_mask, tgt_psulabels = tuple(t.to(args.device) for t in tgt_batch)

            if epoch < 1:
                loss1, loss2, loss3, loss4 = model(
                    src_input_ids=src_input_ids, src_attention_mask=src_input_mask,
                    tgt_input_ids=tgt_input_ids, tgt_attention_mask=tgt_input_mask
                )
                steps(loss2, domain_optimizer, domain_scheduler)
            else:
                for param in model.domain.parameters():
                    param.requires_grad = False
                loss1, loss2, loss3, loss4 = model(
                    src_input_ids=src_input_ids, src_attention_mask=src_input_mask, src_labels=src_label_ids,
                    tgt_input_ids=tgt_input_ids, tgt_attention_mask=tgt_input_mask, pseudo_labels=tgt_psulabels
                )
                total_loss = loss1 + loss3 + (0.1 * loss4 if epoch >= 2 else 0.0)
                steps(total_loss, sent_optimizer, sent_scheduler)

            global_step += 1
            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:%d, Step:%d/%d, Loss: %.4f, %.4f, %.4f, %.4f",
                            epoch, global_step, num_train_steps, loss1.item(), loss2.item(), loss3.item(), loss4.item())
        torch.cuda.empty_cache()


def pseudo_label_to_unlabeled_data(args, model, dataset):
    model.eval()
    target_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    pooled_outputs, out_preds = [], []
    for batch in target_dataloader:
        input_ids, input_mask, _ = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            pooled_output, logits = model(input_ids, input_mask)
            pooled_outputs.append(pooled_output)
            out_preds.append(torch.softmax(logits, dim=1))
    return torch.cat(pooled_outputs, dim=0), torch.cat(out_preds, dim=0)


def evaluate(args, model, eval_dataset):
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.eval_batch_size)
    model.eval()
    out_preds, out_labels = [], []
    for batch in eval_dataloader:
        input_ids, input_mask, label_ids = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            _, _, _, logits = model(src_input_ids=input_ids, src_attention_mask=input_mask)
            predicts = torch.argmax(logits, dim=1)
            out_preds.append(predicts.cpu().numpy())
            out_labels.append(label_ids.cpu().numpy())
    y_true = np.concatenate(out_labels, axis=0)
    y_pred = np.concatenate(out_preds, axis=0)
    acc = accuracy_score(y_true, y_pred)
    logger.info("accuracy: {:.4f}; precision:{:.4f}; recall:{:.4f}; f1:{:.4f}".format(
        acc, precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='macro')
    ))


def load_and_cache_examples(args, trained_model=None, base_feature=False, dataname="train"):
    processor = data_utils.SAProcessor()
    if dataname == "train":
        examples = processor.get_train_examples(args.data_dir)[:args.labeled_num]
    elif dataname == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif dataname == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif dataname == "unlabeled":
        examples = processor.get_target_unexamples(args.data_dir)[:args.unlabeled_num]
    else:
        raise ValueError("Invalid dataname")

    features = data_utils.convert_examples_to_features(examples, args.max_seq_length, args.bert_tokenizer, bert_name='bert', printN=0)
    dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.input_mask for f in features], dtype=torch.long),
        torch.tensor([f.label_id for f in features], dtype=torch.long)
    )
    if base_feature:
        return dataset

    _, pseudo_labels = pseudo_label_to_unlabeled_data(args=args, model=trained_model, dataset=dataset)
    features = data_utils.space_based_convert_examples_to_features(
        examples=examples, max_seq_length=args.max_seq_length,
        pseudo_labels=pseudo_labels, tokenizer=args.bert_tokenizer, bert_name='bert', printN=0
    )
    return TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.input_mask for f in features], dtype=torch.long),
        torch.tensor([f.pseudo_label for f in features], dtype=torch.float)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='bert_base', type=str)
    parser.add_argument("--data_dir", default='../data/amazon/electronics-kitchen', type=str)
    parser.add_argument("--output_dir", default='../run_out/amazon/electronics-kitchen', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument('--eval_logging_steps', type=int, default=100)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument('--logging_global_step', type=int, default=100)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--train_batch_size", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=4, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--labeled_num", default=1600, type=int)
    parser.add_argument("--unlabeled_num", default=4000, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    archive = modelconfig.MODEL_ARCHIVE_MAP.get(args.bert_model, args.bert_model)
    args.bert_tokenizer = SATokenizer(bert_name='bert_base', archive=archive).tokenizer

    trained_model = BertForClassification.from_pretrained(args.output_dir, num_labels=2).to(args.device)
    trained_model.eval()

    train_dataset = load_and_cache_examples(args, base_feature=True, dataname="train")
    unlabeled_dataset = load_and_cache_examples(args, trained_model=trained_model, base_feature=False, dataname="unlabeled")
    dev_dataset = load_and_cache_examples(args, base_feature=True, dataname="dev")
    test_dataset = load_and_cache_examples(args, base_feature=True, dataname="test")
    del trained_model

    model = BertaForClassificationModel.from_pretrained(archive, num_labels=2).to(args.device)
    domainM = BertForDomain.from_pretrained(archive).to(args.device)
    model.domain.load_state_dict(domainM.state_dict())

    if args.do_train:
        train(args, model, train_dataset, unlabeled_dataset, dev_dataset)
        save_path = os.path.join(args.output_dir, "attention_model.pt")
        torch.save(model.state_dict(), save_path)
        logger.info("Saved attention model weights to %s", save_path)

    if args.do_eval:
        evaluate(args, model, train_dataset)
        evaluate(args, model, test_dataset)


if __name__ == "__main__":
    main()
