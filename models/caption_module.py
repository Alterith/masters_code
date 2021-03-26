import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import dropout


class Caption_Module_Architecture(nn.Module):

    def __init__(self,options, device) -> None:

        """This is a captioning Constructor Function
        Args:
        Returns:
            None
        Raises:
            None
        """
        super(Caption_Module_Architecture, self).__init__()

        self.device = device

        # overall project options as a dict
        self.options = options

        # architecture parameters
        self.attention_hidden_size = options["video_feat_dim"]

        
        self.input_dim = options["video_feat_dim"]
        self.hidden_dim = options["encoded_video_feat_dim"]
        self.num_layers = options["num_rnn_layers"]
        #self.num_layers = 10
        self.num_anchors = options["num_anchors"]
        self.batch_size = options["batch_size"]

        # layers 
        ## fix the input dimensions, hidden dim should stay the same input dim is flattened weighted average and the fw bw hidden together but maybe after the cg stuff its just a single 512 vector instead of both layers
        self.lstm_caption_1 = nn.LSTM(2 * self.hidden_dim + 2 * self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout = self.options["rnn_drop"])
        # apply weight norm to lstm layer
        lstm_params = dict(self.lstm_caption_1.named_parameters())
        for name in lstm_params.keys():
            if "bias" not in name:
                nn.utils.weight_norm(self.lstm_caption_1, name)
        
        self.fc_caption = nn.utils.weight_norm(nn.Linear(self.hidden_dim, self.options['vocab_size']))

        # remove batch dim and change batch to 110 max len
        self.fc_attn_1 = nn.utils.weight_norm(nn.Linear(self.input_dim + self.hidden_dim + self.hidden_dim + 2 * self.hidden_dim, self.attention_hidden_size))
        # output dim is 1 because batch is the 110 so we get the 110 values out
        self.fc_attn_2 = nn.utils.weight_norm(nn.Linear(self.attention_hidden_size, 1))

        # visual context gate, for cg to apply will probably need to change output to hidden dim 512 instead of 500
        self.fc_cg_1 = nn.utils.weight_norm(nn.Linear(self.input_dim, 2 * self.hidden_dim))
        # fw bw context gate, 2 is the batch for each fw bw
        # self.fc_cg_2 = nn.utils.weight_norm(nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim))
        ## combined context gate of all features, input, 2 layers and fw + bw so 4x, word embedding, hidden state of previous word

        # combined context gate of all features, input, fw/bw combined after cg2, word embedding, hidden state of previous word (do we just take final layer or both though 2x for now)
        self.fc_cg_3 = nn.utils.weight_norm(nn.Linear(2 * self.hidden_dim + 2 * self.hidden_dim + self.options['word_embed_size'] + 2 * self.hidden_dim , 2 * self.hidden_dim))

        # dim might need to change to -1 not sure, we make the values over total feats sum to 1 instead of the total of each feature, so technically columns must add to one and not rows, so not individual feats but the sum of each individual value over all feats in that position
        self.logsoftmax_norm = nn.LogSoftmax(dim = 1)
        self.softmax_norm = nn.Softmax(dim = 1)

        self.tanh_norm = nn.Tanh()
        
        self.embeddings = nn.utils.weight_norm(nn.Embedding(self.options['vocab_size'], self.options['word_embed_size']))

    # run x in a loop
    def forward(self, feats_ac, temp_proposals_hs_fw, temp_proposals_hs_bw, proposal_caption_fw, proposal_caption_bw):

        if(torch.sum(proposal_caption_fw[0][0]) == 0.):
            idx = min(110, feats_ac.shape[1])
            proposal_caption_fw[0][0][idx - 1] = 1

        end_id = torch.nonzero(proposal_caption_fw[0][0]).view(-1)

        start_id_bw = proposal_caption_bw[0][0][end_id].view(-1)

        start_id = feats_ac.shape[1] -1 - start_id_bw

        if(start_id.nelement() == 0):
            start_id = torch.zeros(1, dtype=torch.int32)

        caption_feats, caption_mask = self.get_video_seq(feats_ac, start_id, end_id, 110)

        caption_feats = caption_feats.to(self.device)

        # should specify number of layers but we for loop through so we dont.
        hidden_state = torch.zeros(self.num_layers, caption_feats.shape[0], self.hidden_dim)

        cell_state = torch.zeros(self.num_layers, caption_feats.shape[0], self.hidden_dim)

        hidden_state = hidden_state.to(self.device)
        cell_state = cell_state.to(self.device)

        hidden = (hidden_state, cell_state)

        fw_hs = temp_proposals_hs_fw[start_id.long()]
        bw_hs = temp_proposals_hs_bw[(feats_ac.shape[1] -1 - end_id).long()]

        #https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.htm
        word_id = torch.Tensor(len(caption_feats))
        word_id = word_id.fill_(self.options['vocab']['<START>'])
        word_id = word_id
        word_id = word_id.long()
        word_id = word_id.to(self.device)
        word_ids = word_id
        word_ids = word_ids.unsqueeze(-1)
        word_conficences = torch.Tensor(len(caption_feats))
        word_conficences = word_conficences.fill_(1.)
        word_conficences = word_conficences.unsqueeze(-1)
        word_conficences = word_conficences.to(self.device)

        fc_word_scores = torch.zeros(caption_feats.shape[0], 1, self.options['vocab_size'], dtype=torch.float32)
        fc_word_scores = fc_word_scores.to(self.device)
        fc_word_scores[:,0,self.options['vocab']['<START>']] = 1.
        # here we loop for the longest possible time
        for i in range(self.options['caption_seq_len']-1):
            
            word_embed = self.embeddings(word_id)
            # print(word_embed)

            hidden_state_reshape = torch.reshape(hidden_state, (-1,self.num_layers*self.hidden_dim))
            
            # concat for attn

            concat_hs = torch.cat((torch.cat((fw_hs, bw_hs), -1), hidden_state_reshape), -1)

            # print(concat_hs.shape)
            concat_hs = concat_hs.unsqueeze(axis=1)
            # print(concat_hs.shape)
            concat_hs = concat_hs.repeat(1,110,1)
            # print(fw_hs.shape, bw_hs.shape, hidden_state_reshape.shape, concat_hs.shape, caption_feats.shape)
            tile = torch.cat((caption_feats, concat_hs), -1)
            tile = tile.float()
            # print(tile.shape)
            
            
            # do attn, pass tile through caption
            attn_out = self.fc_attn_1(tile)
            attn_out = self.tanh_norm(attn_out)
            attn_out = self.fc_attn_2(attn_out)
            attn_out = self.softmax_norm(attn_out)

            attn_weighted_feats = torch.mul(caption_feats, attn_out)

            # returns a [#actions, 1, 500]
            attn_weighted_input = torch.sum(attn_weighted_feats, dim = 1, dtype=torch.float32)
            # print(attn_weighted_input.shape)
            # do cg
            cg_weighted_input = self.fc_cg_1(attn_weighted_input)
            cg_weighted_input = self.tanh_norm(cg_weighted_input)
            # print(cg_weighted_input.shape)

            # cg_weighted_fw_bw = self.fc_cg_2(torch.cat((fw_hs, bw_hs), -1))
            # cg_weighted_fw_bw = self.tanh_norm(cg_weighted_fw_bw)
            cg_weighted_fw_bw = torch.cat((fw_hs, bw_hs), -1)
            # print(cg_weighted_fw_bw.shape)

            # this will need fixing
            # print(cg_weighted_fw_bw.shape, cg_weighted_input.shape, hidden_state_reshape.shape, word_embed.shape)
            cg_final = self.fc_cg_3(torch.cat((torch.cat((torch.cat((cg_weighted_fw_bw, cg_weighted_input), -1), hidden_state_reshape), -1), word_embed), -1))
            cg_final = self.softmax_norm(cg_final)
            # print(cg_final.shape)

            gated_input = (1.0-cg_final) * cg_weighted_input
            gated_hs = cg_final * cg_weighted_fw_bw
            # print(gated_hs.shape)
            gated_caption_input = torch.cat((gated_input, gated_hs), -1)
            gated_caption_input = gated_caption_input.unsqueeze(1)
            # print(gated_caption_input.shape)
            # do caption
            
            caption, cell_hidden = self.lstm_caption_1(gated_caption_input, hidden)

            hidden_state = cell_hidden[0]
            cell_state = cell_hidden[1]
            hidden = cell_hidden
            caption_fc = self.fc_caption(caption)

            # print(caption_fc.shape)
            # print(caption_fc)
            # print(caption_fc.shape)
            caption_word_norm = self.softmax_norm(caption_fc.view(caption_feats.shape[0],-1))
            # # print(caption_fc.sum(dim=1))
            # # print(caption_fc.shape)
            # # quit()
            caption_word_norm = caption_word_norm.unsqueeze(1)

            fc_word_scores = torch.cat((fc_word_scores, caption_fc), dim = 1)

            word_id = torch.argmax(caption_word_norm, dim=-1)
            # print(word_id)

            word_ids = torch.cat((word_ids, word_id), -1)
            word_id = word_id.view(-1)
            # print(word_ids.shape)
            word_confidence = torch.max(caption_word_norm, dim=-1)
            # print(word_confidence)
            # print(word_confidence[0].shape, word_conficences.shape)
            word_conficences = torch.cat((word_conficences, word_confidence[0]), -1)

        return word_ids, fc_word_scores, start_id, end_id #sentences, word_conficences #word_ids, word_confidences



    def caption_eval(self, caption_feats, event_hidden_feats_fw, event_hidden_feats_bw):

        
        with torch.no_grad():
            caption_feats = caption_feats.to(self.device)

            fw_hs = event_hidden_feats_fw
            bw_hs = event_hidden_feats_bw



            # should specify number of layers but we for loop through so we dont.
            hidden_state = torch.zeros(self.num_layers, caption_feats.shape[0], self.hidden_dim)

            cell_state = torch.zeros(self.num_layers, caption_feats.shape[0], self.hidden_dim)

            hidden_state = hidden_state.to(self.device)
            cell_state = cell_state.to(self.device)

            hidden = (hidden_state, cell_state)


            #https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.htm
            word_id = torch.Tensor(len(caption_feats))
            word_id = word_id.fill_(self.options['vocab']['<START>'])
            word_id = word_id
            word_id = word_id.long()
            word_id = word_id.to(self.device)
            word_ids = word_id
            word_ids = word_ids.unsqueeze(-1)
            word_conficences = torch.Tensor(len(caption_feats))
            word_conficences = word_conficences.fill_(1.)
            word_conficences = word_conficences.unsqueeze(-1)
            word_conficences = word_conficences.to(self.device)

            fc_word_scores = torch.zeros(caption_feats.shape[0], 1, self.options['vocab_size'], dtype=torch.float32)
            fc_word_scores = fc_word_scores.to(self.device)
            fc_word_scores[:,0,self.options['vocab']['<START>']] = 1.
            # here we loop for the longest possible time
            for i in range(self.options['caption_seq_len']-1):
                
                word_embed = self.embeddings(word_id)
                # print(word_embed)

                hidden_state_reshape = torch.reshape(hidden_state, (-1,self.num_layers*self.hidden_dim))
                
                # concat for attn

                concat_hs = torch.cat((torch.cat((fw_hs, bw_hs), -1), hidden_state_reshape), -1)

                # print(concat_hs.shape)
                concat_hs = concat_hs.unsqueeze(axis=1)
                # print(concat_hs.shape)
                concat_hs = concat_hs.repeat(1,110,1)
                # print(fw_hs.shape, bw_hs.shape, hidden_state_reshape.shape, concat_hs.shape, caption_feats.shape)
                tile = torch.cat((caption_feats, concat_hs), -1)
                tile = tile.float()
                # print(tile.shape)
                
                
                # do attn, pass tile through caption
                attn_out = self.fc_attn_1(tile)
                attn_out = self.tanh_norm(attn_out)
                attn_out = self.fc_attn_2(attn_out)
                attn_out = self.softmax_norm(attn_out)

                attn_weighted_feats = torch.mul(caption_feats, attn_out)

                # returns a [#actions, 1, 500]
                attn_weighted_input = torch.sum(attn_weighted_feats, dim = 1, dtype=torch.float32)
                # print(attn_weighted_input.shape)
                # do cg
                cg_weighted_input = self.fc_cg_1(attn_weighted_input)
                cg_weighted_input = self.tanh_norm(cg_weighted_input)
                # print(cg_weighted_input.shape)

                #cg_weighted_fw_bw = self.fc_cg_2(torch.cat((fw_hs, bw_hs), -1))
                #cg_weighted_fw_bw = self.tanh_norm(cg_weighted_fw_bw)
                cg_weighted_fw_bw = torch.cat((fw_hs, bw_hs), -1)
                # print(cg_weighted_fw_bw.shape)

                # this will need fixing
                # print(cg_weighted_fw_bw.shape, cg_weighted_input.shape, hidden_state_reshape.shape, word_embed.shape)
                cg_final = self.fc_cg_3(torch.cat((torch.cat((torch.cat((cg_weighted_fw_bw, cg_weighted_input), -1), hidden_state_reshape), -1), word_embed), -1))
                cg_final = self.softmax_norm(cg_final)
                # print(cg_final.shape)

                gated_input = (1.0-cg_final) * cg_weighted_input
                gated_hs = cg_final * cg_weighted_fw_bw
                # print(gated_hs.shape)
                gated_caption_input = torch.cat((gated_input, gated_hs), -1)
                gated_caption_input = gated_caption_input.unsqueeze(1)
                # print(gated_caption_input.shape)
                # do caption
                
                caption, cell_hidden = self.lstm_caption_1(gated_caption_input, hidden)

                hidden_state = cell_hidden[0]
                cell_state = cell_hidden[1]
                hidden = cell_hidden
                caption_fc = self.fc_caption(caption)

                # print(caption_fc.shape)
                # print(caption_fc)
                # print(caption_fc.shape)
                caption_word_norm = self.softmax_norm(caption_fc.view(caption_feats.shape[0],-1))
                # # print(caption_fc.sum(dim=1))
                # # print(caption_fc.shape)
                # # quit()
                caption_word_norm = caption_word_norm.unsqueeze(1)

                fc_word_scores = torch.cat((fc_word_scores, caption_fc), dim = 1)

                word_id = torch.argmax(caption_word_norm, dim=-1)
                # print(word_id)

                word_ids = torch.cat((word_ids, word_id), -1)
                word_id = word_id.view(-1)
                # print(word_ids.shape)
                word_confidence = torch.max(caption_word_norm, dim=-1)
                # print(word_confidence)
                # print(word_confidence[0].shape, word_conficences.shape)
                word_conficences = torch.cat((word_conficences, word_confidence[0]), -1)

        return word_ids, fc_word_scores #sentences, word_conficences #word_ids, word_confidences




    """get video proposal representation (feature sequence), given start end feature ids, all of which are LISTS
    """
    def get_video_seq(self, video_feat_sequence, start_ids, end_ids, max_clip_len):
        #max_clip_len is longest number of features allowed to be apart 110
        
        N = len(start_ids)
        event_video_sequence = torch.empty((0, max_clip_len, self.options['video_feat_dim']), dtype=torch.float32)
        event_video_sequence = event_video_sequence.to(self.device)

        event_video_mask = torch.empty((0, max_clip_len), dtype=torch.int32)

        for event_id in range(0, N):
            start_id = start_ids[event_id]
            end_id = end_ids[event_id] + 1
            
            video_feats = video_feat_sequence[0][start_id:end_id]

            clip_len = end_id - start_id

            this_mask = torch.zeros(max_clip_len)

            if clip_len < max_clip_len:
                this_mask[0:len(video_feats)] = 1
                zero_padding = torch.zeros((max_clip_len - clip_len, self.options['video_feat_dim']), dtype=torch.float32)
                zero_padding = zero_padding.to(self.device)

                video_feats = torch.cat((video_feats, zero_padding), dim = 0)
                
            else:
                this_mask[0:len(this_mask)] = 1
                video_feats = video_feats[:max_clip_len]

            video_feats = torch.unsqueeze(video_feats, dim = 0)
            this_mask = torch.unsqueeze(this_mask, dim = 0)
            # if start_ids[event_id] == end_ids[event_id]:
            #     print(event_video_sequence.shape, video_feats[0].shape)
            try:
                event_video_sequence = torch.cat((event_video_sequence, video_feats), dim=0)
            except Exception as e:
                print(e)
                print(event_video_sequence.shape, video_feats[0].shape, video_feats.shape, clip_len, start_ids, end_ids)
                print(start_id)
                quit()
            event_video_mask = torch.cat((event_video_mask, this_mask), dim=0)

        # [#events, 110, 500]

        return event_video_sequence, event_video_mask
