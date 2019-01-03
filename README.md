# Graininess-aware Deep Feature Learning for Pedestrian Detection
By [Chunze Lin](http://ivg.au.tsinghua.edu.cn/people/Chunze_Lin/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/jiwen_lu/), [Gang Wang](https://damo.alibaba.com/labs/intelligent-transportation), [Jie Zhou](http://www.tsinghua.edu.cn/publish/auen/1713/2011/20110506105532098625469/20110506105532098625469_.html).


### Introduction
GDFL is a graininess-aware deep feature learning based detector for pedestrian detection. We exploit fine-grained details into deep convolutional features for robust pedestrian detection. You can use the code to evaluate the model for pedestrian detection task. For more details, please refer to our [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chunze_Lin_Graininess-Aware_Deep_Feature_ECCV_2018_paper.pdf) and our [poster](http://ivg.au.tsinghua.edu.cn/people/Chunze_Lin/ECCV18_poster_Graininess.pdf).
The GDFL code is based on the implementation of [SSD](https://github.com/weiliu89/caffe/tree/ssd).

### Citing GDFL

Please cite GDFL in your publications if it helps your research:

    @inproceedings{lin2018graininess,
      title={Graininess-aware deep feature learning for pedestrian detection},
      author={Lin, Chunze and Lu, Jiwen and Wang, Gang and Zhou, Jie},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      pages={732--747},
      year={2018}
    }

### Installation
Build the code. 
Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
   ```Shell
   # Modify Makefile.config according to your Caffe installation.
   cp Makefile.config.example Makefile.config
   make -j8
   # Make sure to include $CAFFE-GDFL_ROOT/python to your PYTHONPATH.
   make pycaffe
   ```
### Data Preparation
Download the [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/) dataset and follow the [instructions](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) to extract the annotations and images. Put them in the propor folder. 

### Evaluation
You can download a pre-trained model on Caltech at [here](https://pan.baidu.com/s/1vnUwOCYsDh1aE0QskxdGHg) and store it at 'models/Caltech/'. 
You can evaluate this model on a single image or on the Caltech testing set.
   ```Shell
   # Make sure to include $CAFFE-GDFL_ROOT/python to your PYTHONPATH.
   # if you would like to test the model with a single image, you can do:
   python examples/scripts/single_image_detection.py

   # if you would like to test with Caltech testing set, you can do:
   python examples/scripts/test_caltech.py
   ```

The results of the Caltech testing set will be generated in 'output/' and can be evaluated with [matlab evaluation code](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip).

