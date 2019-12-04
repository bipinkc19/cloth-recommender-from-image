# Cloth Recommendation System

Convolutional Neural Network for recommending cloths


### Creating a custom model to train still remaining due to lack of resources for now

## Output from a VGG19 (pre-trained)

### Input image

![a](./images/eg2.png) <br>

### Returned similar images

![a](./images/res2a.png) <br>
![a](./images/res2b.png) <br>
![a](./images/res2c.png) <br>

### Input image

![a](./images/eg3.png) <br>

### Returned similar images

![a](./images/res3a.png) <br>
![a](./images/res3b.png) <br>
![a](./images/res3c.png) <br>

### Input image

![a](./images/eg1.png) <br>

### Returned similar images

![a](./images/res1a.png) <br>
![a](./images/res1b.png) <br>
![a](./images/res1c.png) <br>

## Install the dependencies.

```bash
$ pip install -r requirements.txt
```

## Augment images
```bash
$ python augment_images.py
```

## To train the model.
```bash
$ python train.py
```

## For tensorboard vizualization while training.
```bash
$ tensorboard --logdir=tensorboard_logs --host=localhost --port=8088
```

## Note:

Different model require different size if images. This can be edited in the `tf_records.py` in resize opereation.
