import torch
import torch.nn as nn
import numpy as np


class Temporal_Proposal_Architecture(nn.Module):

    def __init__(self, options) -> None:

        """This is a captioning Constructor Function
        Args:
        Returns:
            None
        Raises:
            None
        """
        super(Temporal_Proposal_Architecture, self).__init__()

        # overall project options as a dict
        self.options = options

        # architecture parameters
        self.input_dim = options["video_feat_dim"]
        self.hidden_dim = options["encoded_video_feat_dim"]
        self.num_layers = 1
        #self.num_layers = 10
        self.num_anchors = options["num_anchors"]
        self.batch_size = options["batch_size"]

        # layers
        self.lstm_proposal_fw = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers, batch_first=True, dropout=options['rnn_drop'])

        self.lstm_proposal_bw = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers, batch_first=True, dropout=options['rnn_drop'])


        self.fc_fw = nn.Linear(self.hidden_dim, self.num_anchors)

        self.fc_bw = nn.Linear(self.hidden_dim, self.num_anchors)


        self.activation_fw = nn.Sigmoid()
        self.activation_bw = nn.Sigmoid()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, data_fw, data_bw):

        """This is a temporal action proposal forward pass function
        Args:
            x (torch tensor): The input video clips
        Returns:
            torch tensor: Representing the processed clip 'x'
        Raises:
            None
        """


        # initial state for lstm
        hidden_state_fw = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        hidden_state_fw = hidden_state_fw.to(self.device)
        cell_state_fw = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        cell_state_fw = cell_state_fw.to(self.device)
        fw_hs_cc = (hidden_state_fw, cell_state_fw)

        # # the outputs and hidden state for each time step
        # fw_out_cat = None
        # fw_hs_cc_cat = None

        hidden_state_bw = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        hidden_state_bw = hidden_state_bw.to(self.device)
        cell_state_bw = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        cell_state_bw = cell_state_bw.to(self.device)
        bw_hs_cc = (hidden_state_bw, cell_state_bw)

        # # the outputs and hidden state for each time step
        # bw_out_cat = None
        # bw_hs_cc_cat = None

        # # loop to get hidden state for all time steps
        # for seq_num in range(data_fw.shape[1]):
        #     fw_out, fw_hs_cc = self.lstm_proposal_fw(data_fw[:,seq_num,:].unsqueeze(0), fw_hs_cc)

        #     # concat or initialize the tensors which store all the data
        #     if seq_num == 0:
        #         fw_out_cat = fw_out
        #         # convert returned tuple to list else it is immutable and we cant concat the subsequent sequence items
        #         fw_hs_cc_cat = list(fw_hs_cc)
        #     else:
        #         fw_out_cat = torch.cat((fw_out_cat, fw_out), dim = 1)
        #         fw_hs_cc_cat[0] = torch.cat((fw_hs_cc_cat[0], fw_hs_cc[0]), dim = 1)
        #         fw_hs_cc_cat[1] = torch.cat((fw_hs_cc_cat[1], fw_hs_cc[1]), dim = 1)

        fw_out, fw_hs_cc = self.lstm_proposal_fw(data_fw, fw_hs_cc)
        fw_out_reshape = fw_out.view(-1, self.hidden_dim)


        # # loop to get hidden state for all time steps
        # for seq_num in range(data_bw.shape[1]):
        #     bw_out, bw_hs_cc = self.lstm_proposal_bw(data_bw[:,seq_num,:].unsqueeze(0), bw_hs_cc)

        #     # concat or initialize the tensors which store all the data
        #     if seq_num == 0:
        #         bw_out_cat = bw_out
        #         # convert returned tuple to list else it is immutable and we cant concat the subsequent sequence items
        #         bw_hs_cc_cat = list(bw_hs_cc)
        #     else:
        #         bw_out_cat = torch.cat((bw_out_cat, bw_out), dim = 1)
        #         bw_hs_cc_cat[0] = torch.cat((bw_hs_cc_cat[0], bw_hs_cc[0]), dim = 1)
        #         bw_hs_cc_cat[1] = torch.cat((bw_hs_cc_cat[1], bw_hs_cc[1]), dim = 1)

        bw_out, bw_hs_cc = self.lstm_proposal_bw(data_bw, bw_hs_cc)
        bw_out_reshape = bw_out.view(-1, self.hidden_dim)

        logit_output_fw = self.fc_fw(fw_out_reshape)
        logit_output_bw = self.fc_bw(bw_out_reshape)

        # logit_output_fw = self.activation_fw(logit_output_fw)
        # logit_output_bw = self.activation_bw(logit_output_bw)

        return logit_output_fw, logit_output_bw, fw_out_reshape, bw_out_reshape