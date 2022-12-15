import torch
from torch_geometric.data import InMemoryDataset, download_url

from process_mining_services import log_to_batch_files, import_xes, batch_files_to_datalist

"""
PyTorch Lightning Dataset based on batched process log mining
"""
class ProcessDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['log.xes']

    @property
    def processed_file_names(self):
        return ['finished_processing_flag']

    def download(self):
        # Download to `self.raw_dir`.
        download_url("https://cdn.gereonelvers.com/datasets/log.xes", self.raw_dir)

    def process(self):
        # Read process-datasets into huge `Data` list.
        data_list = []

        # TODO: Figure out what to do about pre-processing.
        # if self.pre_filter is not None:
        #     data_list = [process-datasets for process-datasets in data_list if self.pre_filter(process-datasets)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(process-datasets) for process-datasets in data_list]

        # Steps:
        log = import_xes(self.raw_dir + '/log.xes')  # Import log
        log_to_batch_files(log, traces_per_batch=1, batches_receptive_field=1, first_batch_size=1, path=self.processed_dir)  # Convert log to batch files
        data_list = batch_files_to_datalist(self.processed_dir)  # Convert batch files to dataset

        # Save process-datasets to `self.processed_dir`.
        # process-datasets, slices = self.collate(data_list)
        # torch.save((process-datasets, slices), self.processed_paths[0])

        # Set a flag to indicate that the log has been processed.
        flag = open(self.processed_dir + "/finished_processing.flag", 'w')
        flag.write("Finished processing.")
        flag.close()
