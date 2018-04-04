# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from helpers import visualize, nnet_helpers
from torch.autograd import Variable
from modules import cls_sparse_skip_filt as s_s_net
from losses import loss_functions
from helpers import iterative_inference as it_infer
import torch.nn as nn
import os

# torch.set_num_threads(16)


def main(training, apply_sparsity):
    """
        The main function to train and test.
    """
    # Reproducible results
    np.random.seed(218)
    torch.manual_seed(218)
    # torch.cuda.manual_seed(218)
    # Torch model
    torch.set_default_tensor_type('torch.FloatTensor')

    # Analysis
    wsz = 2049   # Window-size
    Ns = 4096    # FFT size
    hop = 384    # Hop size
    fs = 44100   # Sampling frequency

    # Parameters
    B = 16                      # Batch-size
    T = 60                      # Length of the sequence
    N = 2049                    # Frequency sub-bands to be processed
    F = 744                     # Frequency sub-bands for encoding
    L = 10                      # Context parameter (2*L frames will be removed)
    epochs = 20                 # Epochs
    init_lr = 1e-4              # Initial learning rate
    mnorm = 0.5	                # L2-based norm clipping
    mask_loss_threshold = 1.5   # Scalar indicating the threshold for the time-frequency masking module
    good_loss_threshold = 0.25  # Scalar indicating the threshold for the source enhancment module

    # Data (Predifined by the DSD100 dataset and the non-instumental/non-bleeding stems of MedleydB)
    totTrainFiles = 50
    numFilesPerTr = 2

    print('------------   Building model   ------------')
    encoder = s_s_net.BiGRUEncoder(B, T, N, F, L)
    decoder = s_s_net.Decoder(B, T, N, F, L, infr=True)
    sp_decoder = s_s_net.SparseDecoder(B, T, N, F, L)
    source_enhancement = s_s_net.SourceEnhancement(B, T, N, F, L)
    source_enhancement_shunit = s_s_net.SourceEnhancement(B, T, N, F, L)

    encoder.train(mode=True)
    decoder.train(mode=True)
    sp_decoder.train(mode=True)
    source_enhancement.train(mode=True)
    source_enhancement_shunit.train(mode=True)

    if torch.has_cudnn:
        print('------------   CUDA Enabled   --------------')
        encoder.cuda()
        decoder.cuda()
        sp_decoder.cuda()
        source_enhancement.cuda()
        source_enhancement_shunit.cuda()

    print('-------  Loading pre-trained model   -------')
    print('-------  Loading inference weights  -------')
    encoder.load_state_dict(
        torch.load('results/results_inference/torch_sps_encoder.pytorch', map_location=lambda storage, loc: storage))
    decoder.load_state_dict(
        torch.load('results/results_inference/torch_sps_decoder.pytorch', map_location=lambda storage, loc: storage))
    sp_decoder.load_state_dict(
        torch.load('results/results_inference/torch_sps_sp_decoder.pytorch', map_location=lambda storage, loc: storage))
    # source_enhancement.load_state_dict(
    #     torch.load('results/results_inference/torch_sps_se.pytorch', map_location=lambda storage, loc: storage))
    print('-------------      Done        -------------')

    for x in encoder.parameters():
        # print(x.requires_grad)
        x.requires_grad = False
    for x in decoder.parameters():
        x.requires_grad = False
    for x in sp_decoder.parameters():
        x.requires_grad = False
    for x in source_enhancement.parameters():
        x.requires_grad = False
    source_enhancement.ffSe_enc = nn.Linear(N, N // 2)
    source_enhancement.ffSe_dec = nn.Linear(N // 2, N)


    # Defining objectives
    rec_criterion = loss_functions.kullback_leibler                 # Reconstruction criterion

    # optimizer = optim.Adam(list(encoder.parameters()) +
    #                        list(decoder.parameters()) +
    #                        list(sp_decoder.parameters()) +
    #                        list(source_enhancement.parameters()),
    #                        lr=init_lr
    #                        )
    optimizer = optim.Adam(list(source_enhancement.parameters()),
                           lr=init_lr
                           )

    for x in encoder.parameters():
        print(x.requires_grad)
    for x in decoder.parameters():
        print(x.requires_grad)
    for x in sp_decoder.parameters():
        print(x.requires_grad)
    for x in source_enhancement.parameters():
        print(x.requires_grad)

    if training:
        win_viz, winb_viz = visualize.init_visdom()
        batch_loss = []
        # Over epochs
        batch_index = 0
        for epoch in range(epochs):
            print('Epoch: ' + str(epoch+1))
            epoch_loss = []
            # Over the set of files
            for index in range(totTrainFiles//numFilesPerTr):
                # Get Data
                ms, vs = nnet_helpers.get_data_shunit(index+1, numFilesPerTr, wsz, Ns, hop, T, L, B)
                # print(ms.shape)
                # Shuffle data
                shf_indices = np.random.permutation(ms.shape[0])
                ms = ms[shf_indices]
                vs = vs[shf_indices]
                # print(ms.shape)
                # Over batches
                for batch in tqdm(range(ms.shape[0]//B)):
                    # Mixture to Singing voice
                    # print(ms[batch * B: (batch+1)*B, :, :].shape)
                    H_enc = encoder(ms[batch * B: (batch+1)*B, :, :])
                    # print(H_enc.shape)
                    # Iterative inference
                    H_j_dec = it_infer.iterative_recurrent_inference(decoder, H_enc,
                                                                     criterion=None, tol=1e-3, max_iter=10)
                    # print(H_j_dec.shape)
                    vs_hat_b = sp_decoder(H_j_dec, ms[batch * B: (batch+1)*B, :, :])[0]
                    # print(vs_hat_b.shape)
                    vs_hat_b_filt = source_enhancement(vs_hat_b)
                    # print(vs_hat_b_filt.shape)
                    # exit()

                    # Loss
                    if torch.has_cudnn:
                        loss = rec_criterion(Variable(torch.from_numpy(vs[batch * B: (batch+1)*B, L:-L, :]).cuda()),
                                         vs_hat_b_filt)

                        loss_mask = rec_criterion(Variable(torch.from_numpy(vs[batch * B: (batch+1)*B, L:-L, :]).cuda()),
                                         vs_hat_b)

                        if loss_mask.data[0] >= mask_loss_threshold and loss.data[0] >= good_loss_threshold:
                            loss += loss_mask

                    else:
                        loss = rec_criterion(Variable(torch.from_numpy(vs[batch * B: (batch+1)*B, L:-L, :])),
                                         vs_hat_b_filt)

                        loss_mask = rec_criterion(Variable(torch.from_numpy(vs[batch * B: (batch+1)*B, L:-L, :])),
                                         vs_hat_b)

                        if loss_mask.data[0] >= mask_loss_threshold and loss.data[0] >= good_loss_threshold:
                            loss += loss_mask

                    # Store loss for display and scheduler
                    batch_loss += [loss.data[0]]
                    epoch_loss += [loss.data[0]]

                    # Sparsity term
                    if apply_sparsity:
                        sparsity_penalty = torch.sum(torch.abs(torch.diag(sp_decoder.ffDec.weight.data))) * 1e-2 +\
                                           torch.sum(torch.pow(source_enhancement.ffSe_dec.weight, 2.)) * 1e-4

                        loss += sparsity_penalty

                        winb_viz = visualize.viz.line(X=np.arange(batch_index, batch_index+1),
                             Y=np.reshape(sparsity_penalty.data[0], (1,)),
                             win=winb_viz, update='append')

                    optimizer.zero_grad()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm(list(encoder.parameters()) +
                                                  list(decoder.parameters()) +
                                                  list(sp_decoder.parameters()) +
                                                  list(source_enhancement.parameters()),
                                                  max_norm=mnorm, norm_type=2)
                    optimizer.step()
                    # Update graphs
                    win_viz = visualize.viz.line(X=np.arange(batch_index, batch_index+1),
                                                 Y=np.reshape(batch_loss[batch_index], (1,)),
                                                 win=win_viz, update='append')
                    batch_index += 1

            if (epoch+1) % 1 == 0:
                print('------------   Saving model   ------------')
                # torch.save(encoder.state_dict(), 'results/torch_sps_encoder_shunit_' + str(epoch+1)+'.pytorch')
                # torch.save(decoder.state_dict(), 'results/torch_sps_decoder_shunit_' + str(epoch+1)+'.pytorch')
                # torch.save(sp_decoder.state_dict(), 'results/torch_sps_sp_decoder_shunit_' + str(epoch+1)+'.pytorch')
                torch.save(source_enhancement.state_dict(), 'results/torch_sps_se_shunit_' + str(epoch+1)+'.pytorch')
                torch.save(source_enhancement.state_dict(), 'results/results_inference/torch_sps_se_shunit.pytorch')
                print('------------       Done       ------------')
    else:
        print('-------  Loading pre-trained model   -------')
        print('-------  Loading inference weights  -------')
        encoder.load_state_dict(torch.load('results/results_inference/torch_sps_encoder.pytorch', map_location=lambda storage, loc: storage))
        decoder.load_state_dict(torch.load('results/results_inference/torch_sps_decoder.pytorch', map_location=lambda storage, loc: storage))
        sp_decoder.load_state_dict(torch.load('results/results_inference/torch_sps_sp_decoder.pytorch', map_location=lambda storage, loc: storage))
        source_enhancement.load_state_dict(torch.load('results/results_inference/torch_sps_se.pytorch', map_location=lambda storage, loc: storage))
        source_enhancement_shunit.load_state_dict(torch.load('results/results_inference/torch_sps_se_shunit.pytorch', map_location=lambda storage, loc: storage))
        print('-------------      Done        -------------')

    return encoder, decoder, sp_decoder, source_enhancement, source_enhancement_shunit


if __name__ == '__main__':
    os.chdir('/home/shunith/MusicSourceSeparation')
    training = True         # Whether to train or test the trained model (requires the optimized parameters)
    apply_sparsity = True    # Whether to apply a sparse penalty or not

    sfiltnet = main(training, apply_sparsity)

    print('-------------     BSS-Eval     -------------')
    nnet_helpers.test_eval_shunit(sfiltnet, 16, 60, 4096, 10, 2049, 384)
    print('-------------       Done       -------------')
    # print('-------------     DNN-Test     -------------')
    # nnet_helpers.test_nnet(sfiltnet, 60, 10*2, 2049, 4096, 384, 16)
    # print('-------------       Done       -------------')

# EOF
