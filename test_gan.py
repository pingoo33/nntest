import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image

from data.interface.data_manager import DataManager
from model.interface.model_manager import ModelManager
from model.threshold_manager import ThresholdManager
from test.fc.boundary_coverage import NeuronBoundaryCoverage
from test.fc.k_multisection_coverage import KMultisectionCoverage
from test.fc.threshold_coverage import ThresholdCoverage
from test.fc.top_k_coverage import TopKCoverage


class TestGAN:
    def __init__(self, data_manager: DataManager, model_manager: ModelManager):
        self.data_manager = data_manager
        self.model_manager = model_manager

        self.dir_list = ['acgan', 'began', 'dagan', 'cgan', 'dcgan', 'ebgan', 'lsgan', 'relativistic_gan', 'sgan',
                         'wgan_gp']
        # self.dir_list = ['dagan']

    def __data_process(self, coverage_set, target_data, target_label, name):
        pbar = tqdm(range(len(target_data)), total=len(target_data))
        for i in pbar:
            before_output = int(target_label[i][0])
            generated_data = target_data[i]

            after_output = self.model_manager.get_prob(generated_data)
            after_output = np.argmax(after_output)
            self.data_manager.update_sample(before_output, after_output)

            for coverage in coverage_set:
                if not (generated_data is None):
                    coverage.update_features(generated_data)
                    coverage.update_graph(self.data_manager.get_num_samples())

            pbar.set_description("name: %s, samples: %d, advs: %d"
                                 % (name, self.data_manager.get_num_samples(), self.data_manager.get_num_advs()))
        for coverage in coverage_set:
            coverage.update_frequency_graph()
            coverage.display_graph(name)
            coverage.display_frequency_graph(name)
            coverage.display_stat(name)

        for coverage in coverage_set:
            _, result = coverage.calculate_coverage()
            print("name: %s, %s : %.5f" % (name, coverage.get_name(), result))

        self.data_manager.reset_sample()

    def __load_data(self):
        data_list = []
        label_list = []

        for dir in self.dir_list:
            path = './images/' + dir + '/gen'
            file_names = os.listdir(path)

            datas = []
            labels = []
            for file in file_names:
                img_path = os.path.join(path, file)
                img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
                img_tensor = image.img_to_array(img)
                img_tensor = img_tensor.reshape((28, 28, 1))
                img_tensor /= 255

                label = file.split('.')[0]
                label = label.split('_')[1]
                datas.append(img_tensor)
                labels.append(label)
            data_list.append(datas)
            label_list.append(labels)

        return data_list, label_list

    def fc_test(self, threshold_tc, sec_kmnc, size_tkc):
        self.model_manager.load_model()
        target_data, target_label = self.__load_data()
        _, other_layers = self.model_manager.get_fc_layer()

        for layer in other_layers:
            threshold_manager = ThresholdManager(self.model_manager, layer, self.data_manager.x_train)
            for i in range(len(self.dir_list)):
                coverage_set = [ThresholdCoverage(layer, self.model_manager, threshold_tc),
                                KMultisectionCoverage(layer, self.model_manager, threshold_manager, sec_kmnc),
                                NeuronBoundaryCoverage(layer, self.model_manager, threshold_manager),
                                TopKCoverage(layer, self.model_manager, size_tkc)]
                # coverage_set = [KMultisectionCoverage(layer, self.model_manager, threshold_manager, sec_kmnc)]

                self.__data_process(coverage_set, target_data[i], target_label[i], self.dir_list[i])
