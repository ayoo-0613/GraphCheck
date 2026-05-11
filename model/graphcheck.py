import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from model.gnn import load_gnn_model


# -----------------------------------------------------------
# Part of this code is adapted from the G-Retriever project:
# https://github.com/XiaoxinHe/G-Retriever
# He et al. (2024), "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering"
# arXiv:2402.07630
# -----------------------------------------------------------


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphCheck(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        revision = "main"
        if torch.cuda.is_available():
            runtime_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            runtime_device = torch.device("mps")
        else:
            runtime_device = torch.device("cpu")

        print(f"Loading LLM on device: {runtime_device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path,
            use_fast=False,
            revision=revision,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
        self.tokenizer.padding_side = 'left'
        self.rating_labels = ["1", "2", "3", "4", "5"]
        self.rating_token_ids = []
        for rating in self.rating_labels:
            ids = self.tokenizer(
                rating,
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]
            assert len(ids) >= 1
            self.rating_token_ids.append(ids[-1])

        if runtime_device.type == "cuda":
            num_devices = torch.cuda.device_count()
            max_memory = {}
            for i in range(num_devices):
                total_memory = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                max_memory[i] = f"{max(total_memory - 2, 2)}GiB"

            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                device_map="auto",
                revision=revision,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                revision=revision,
                trust_remote_code=True,
            )
            model = model.to(runtime_device)
        
        # Freezing LLM
        for name, param in model.named_parameters():
            param.requires_grad = False

        if runtime_device.type == "cuda":
            model.gradient_checkpointing_enable()

        self.model = model
    
        print('Finish loading LLM!!!')

        self.word_embedding = self.model.get_input_embeddings()
        embedding_device = self.word_embedding.weight.device

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(embedding_device)
        
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.word_embedding.weight.shape[1]),
        ).to(embedding_device)

        self.embed_dim = self.word_embedding.weight.shape[1]
        self.gnn_output = args.gnn_hidden_dim
              

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    # autocast
    def maybe_autocast(self, dtype=torch.bfloat16):
        if self.word_embedding.weight.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()
    
    # Graph Encoding
    def encode_graphs(self, data):
        device = self.word_embedding.weight.device
        claim_kg = data['claim_kg'].to(device)
        doc_kg = data['doc_kg'].to(device)
        
        claim_n_embeds, _ = self.graph_encoder(claim_kg.x, claim_kg.edge_index.long(), claim_kg.edge_attr)   
        doc_n_embeds, _ = self.graph_encoder(doc_kg.x, doc_kg.edge_index.long(), doc_kg.edge_attr)
  
        if claim_kg.batch is not None:  
            claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce='mean')  
        else:  
            claim_embeds = claim_n_embeds.mean(dim=0, keepdim=True)
            
        if doc_kg.batch is not None:  
            doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce='mean')  
        else:  
            doc_embeds = doc_n_embeds.mean(dim=0, keepdim=True)

        return claim_embeds, doc_embeds

    def forward(self, data):
        # prompt texts and corresponding labels
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        labels = self.tokenizer(data["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        device = self.word_embedding.weight.device
        bos_input_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(device)
        bos_embeds = self.word_embedding(bos_input_ids)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device=device)).unsqueeze(0)

        # encode graphs of claims and graphs of documents separately
        claim_embeds, doc_embeds = self.encode_graphs(data)

        # projection
        claim_embeds = self.projector(claim_embeds) 
        doc_embeds = self.projector(doc_embeds)

        batch_size = len(data['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids   
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=device))
            
            # print(f"claim_embeds shape: {claim_embeds.shape}")
            # print(f"doc_embeds shape: {doc_embeds.shape}")
            
            # if claim_embeds or doc_embeds is null
            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = torch.zeros(self.embed_dim, device=device).unsqueeze(0)

            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = torch.zeros(self.embed_dim, device=device).unsqueeze(0)
            
            inputs_embeds = torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # padding
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)
        attention_mask = torch.tensor(batch_attention_mask, device=device)
        label_input_ids = torch.tensor(batch_label_input_ids, device=device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def is_amazon_rating_batch(self, data):
        datasets = data.get("dataset", [])
        if isinstance(datasets, str):
            datasets = [datasets]
        if any(str(dataset).startswith("Amazon") for dataset in datasets):
            return True

        task_types = data.get("task_type", [])
        if isinstance(task_types, str):
            task_types = [task_types]
        if any(str(task_type) == "rating" for task_type in task_types):
            return True

        labels = data.get("label", [])
        if isinstance(labels, str):
            labels = [labels]
        rating_values = set(self.rating_labels)
        texts = data.get("text", [])
        if isinstance(texts, str):
            texts = [texts]
        has_rating_prompt = any("predict the rating" in str(text) for text in texts)
        return bool(labels) and all(str(label) in rating_values for label in labels) and has_rating_prompt

    def inference(self, data):
        if self.is_amazon_rating_batch(data):
            return self.inference_rating(data)

        # encode prompt
        texts = self.tokenizer(data["text"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        device = self.word_embedding.weight.device
        bos_input_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(device)
        bos_embeds = self.word_embedding(bos_input_ids)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device=device)).unsqueeze(0)

        # encode graphs
        claim_embeds, doc_embeds = self.encode_graphs(data)
        # projection
        claim_embeds = self.projector(claim_embeds)
        doc_embeds = self.projector(doc_embeds)
        
        # data['id'] = [data['id']] if isinstance(data['id'], int) else data['id']
        batch_size = len(data['id'])

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=device))
            
            # if claim_embeds or doc_embeds is null
            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = torch.zeros(self.embed_dim, device=device).unsqueeze(0)
            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = torch.zeros(self.embed_dim, device=device).unsqueeze(0)
            
            inputs_embeds = torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # padding
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)
        attention_mask = torch.tensor(batch_attention_mask, device=device)
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(pred)
        return {'id': data['id'],
                'pred': pred,
                'label': data['label'],
                'text': data['text']}

    def inference_rating(self, data):
        texts = self.tokenizer(data["text"], add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        device = self.word_embedding.weight.device
        bos_input_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(device)
        bos_embeds = self.word_embedding(bos_input_ids)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device=device)).unsqueeze(0)

        claim_embeds, doc_embeds = self.encode_graphs(data)
        claim_embeds = self.projector(claim_embeds)
        doc_embeds = self.projector(doc_embeds)

        batch_size = len(data['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=device))

            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = torch.zeros(self.embed_dim, device=device).unsqueeze(0)
            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = torch.zeros(self.embed_dim, device=device).unsqueeze(0)

            inputs_embeds = torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)
        attention_mask = torch.tensor(batch_attention_mask, device=device)
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

        next_token_logits = outputs.logits[:, -1, :]
        rating_token_ids = torch.tensor(self.rating_token_ids, device=next_token_logits.device)
        rating_logits = torch.index_select(next_token_logits, dim=-1, index=rating_token_ids)
        pred_indices = rating_logits.argmax(dim=-1)
        pred_ratings = [str(int(i.item()) + 1) for i in pred_indices]
        print(pred_ratings)

        return {'id': data['id'],
                'pred': pred_ratings,
                'label': data['label'],
                'text': data['text']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
