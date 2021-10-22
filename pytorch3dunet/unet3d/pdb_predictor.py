import os
import h5py
import numpy as np
import torch
from pathlib import Path
from pytorch3dunet.datasets.pdb import PDBDataset
from pytorch3dunet.unet3d.utils import get_logger
import prody

logger = get_logger('UNetPredictor')

class _AbstractPredictorPdb:
    def __init__(self, model, output_dir, config, **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def volume_shape(dataset):
        if dataset.ndim == 3:
            return dataset.shape
        else:
            return dataset.shape[1:]

    @staticmethod
    def get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def __call__(self, test_loader):
        raise NotImplementedError


class PdbPredictor(_AbstractPredictorPdb):
    def __init__(self, model, output_dir, config, **kwargs):
        self.saveh5 = kwargs.get('save_h5', False)
        super().__init__(model, output_dir, config, **kwargs)

    def __call__(self, test_loader):

        assert isinstance(test_loader.dataset, PDBDataset)
        dataset: PDBDataset = test_loader.dataset

        name = dataset.name
        logger.info(f"Processing '{name}'...")

        output_folder = Path(self.output_dir) / name
        os.makedirs(output_folder, exist_ok=True)
        output_file_h5 = str(output_folder / "grids.h5")
        output_file_pdb = str(output_folder / "pocket_pred.pdb")

        out_channels = self.config['model'].get('out_channels')

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(test_loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = self.volume_shape(test_loader.dataset)
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        slice_builder_config = self.config['loaders']['test']['slice_builder']
        if slice_builder_config['name'] == 'TrivialSliceBuilder':
            patch_halo = None
            logger.info(f'Skipping halo validation because of Trivial Slice Builder (TO BE IMPLEMENTED)')
        else:
            patch_halo = self.predictor_config.get('patch_halo', (4, 8, 8))
            self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
            logger.info(f'Using patch_halo: {patch_halo}')

        # create destination H5 file
        if self.saveh5:
            h5_output_file = h5py.File(output_file_h5, 'w')
        else:
            h5_output_file = None

        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape, output_heads)

        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in test_loader:
                # send batch to device
                batch = batch.to(device)
                logger.info(f"Batch shape: {batch.shape}")

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):
                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        logger.info(f'Saving predictions for slice:{index}...')

                        u_prediction, u_index = pred, index

                        # accumulate probabilities into the output prediction array
                        prediction_map[u_index] += u_prediction
                        # count voxel visits for normalization
                        normalization_mask[u_index] += 1

        # save results
        logger.info(f'Saving predictions to: {output_file_pdb}')
        self._save_results(prediction_maps, normalization_masks, output_heads, test_loader.dataset,
                           output_file_pdb, output_h5_file=h5_output_file)
        # close the output H5 file
        if h5_output_file is not None:
            h5_output_file.close()

    def _allocate_prediction_maps(self, output_shape, output_heads):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, dataset : PDBDataset, output_pdb_path,
                      output_h5_file=None):
        pdbData = dataset.pdbDataHandler

        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_datasets = self.get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask

            if output_h5_file is not None:
                output_h5_file.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")

            if True:
                s = pdbData.makePdbPrediction(prediction_map[0])
                if len(s) == 0:
                    open(f'{output_pdb_path}.empty', 'a').close()
                else:
                    prody.writePDB(output_pdb_path, s)

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"
