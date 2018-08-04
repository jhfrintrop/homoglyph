# Detecting Homoglyph Attacks with a Siamese Neural Network
This is a fork of [endgameinc/homoglyph](https://github.com/endgameinc/homoglyph). **It requires Python 3**

This repository adds a `CNN` class that wraps the functionality and a `Archive` class that creates a zip archive of the neural network and needed metadata. 

**Python 3 required!**

**This code was written and tested with Python 3.6**


## Dependencies

First you need to install the dependencies from [requirements.txt](requirements.txt), preferably with `pip`.

It is recommended to use a virtual environment like [virtualenv](https://virtualenv.pypa.io/en/stable/).

```
pip install -r requirements.txt
```


### Arial.ttf
The sample models were trained with Arial.ttf and the original repository and the paper also used Arial.ttf.


#### Mac

You can view this for getting Arial.ttf: https://support.apple.com/guide/font-book/install-and-validate-fonts-fntbk1000/mac


#### Ubuntu

You can install ttf-mscorefonts-installer


#### Windows

Look in `c:\Windows\Fonts`


## Run the code

**Python 3 required!**

`cli.py` is a simple command line utility. Run it like

```
python cli.py
```


## CNN

```
$ python cli.py --help
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

  Work with a siamese convolutional neural network

Options:
  --loglevel LOGLEVEL      The minimal loglevel to log.  [default: warning]
  --log-format LOG-FORMAT  The log format or the name of the log format.
                           [default: [%(asctime)s] [%(levelname)-8s]
                           %(processName)s:%(process)d %(name)s: %(message)s]
  --help                   Show this message and exit.
  --version                Show the version and exit.

Commands:
  archive  Manage archives
  predict  Predict the similarity of pairs of strings
  train    Train or retrain a neural network
```


### Train

```
$ python cli.py train --help
Usage: cli.py train [OPTIONS] FONT DATA [ARCHIVE]

  Train or retrain a neural network

Options:
  --font-size INTEGER RANGE       The font size  [default: 10]
  --image-size <INTEGER INTEGER>...
                                  The size of the image  [default: 150, 12]
  --text-location <INTEGER INTEGER>...
                                  The starting location of the text  [default:
                                  0, 0]
  --max-epochs INTEGER RANGE      The maximum number of epochs to train
                                  [default: 25]
  --verbose / --quiet             If the progress of the training steps should
                                  be shown  [default: False]
  --model PATH                    The path of the archive of the model which
                                  should be retrained
  --test                          If this is a test run with very limited
                                  dataset size  [default: False]
  --fast                          If this is a fast run with limited dataset
                                  size  [default: False]
  --version INTEGER RANGE         Which archive version to use
  --help                          Show this message and exit.
```

To train you need to specify the font to use (the path to the font file) and the path to the data file.

The data must be a pickled dictionary of the following structure:
Keys `train`, `validate` and `test`.
Each key holds a list of 3 tuples `(str, str, float)`: two strings to compare and an expected similarity.

```python
{
    'train': [('string a', 'string b', 0.1), ...],
    'validate': [('string a', 'string b', 0.1), ...],
    'test': [('string a', 'string b', 0.1), ...],
}
```

The `--test` mode drastically reduces the size of each dataset and the maximum number of epochs to train and should be used to test if the program still works.

The `--fast` mode also reduces the size of each dataset and the max. epochs but not as much. It is used to get faster but worse results than a full training.

For example use the following to train domain names for test purposes:

```
$ python cli.py train Arial.ttf data\domain.pkl model.zip --test
Using TensorFlow backend.
TEST, reducing data and max. epochs
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
```

The `Using TensorFlow backend.` and `I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2` messages are only informational messages and might vary.


#### Retrain

If you want to retrain a neural network you have to specify the path to the archive of the model to retrain with `--model path/to/model.zip`.

**This currently only works with archives with version 2!**


### Predict

```
$ python cli.py predict --help
Usage: cli.py predict [OPTIONS] MODEL [DOMAINS]...

  Predict the similarity of pairs of strings

Options:
  --threshold THRESHOLD  If given display only results where the prediction is
                         lower or equal this threshold
  --help                 Show this message and exit.
```

To predict some similarities you need to specify the model to use and one or more string pairs as single strings.

```
$ python cli.py predict model.zip googIe.com,google.com "facebook.com, google.com"
Using TensorFlow backend.
I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
googIe.com ~ google.com = 0.05477425083518028
facebook.com ~ google.com = 0.8001943230628967
```

The `Using TensorFlow backend.` and `I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2` messages are only informational messages and might vary.

They come from keras and tensorflow and can't be disabled unfortunately.


## Archive

The archives are just ZIPs and should have the `.zip` extension.

They contain metadata in a `meta.json`:
* the version of the archive
* a build timestamp in unix time (seconds since 1 January 1970 00:00:00 UTC)
* data about the used font
    - the name of the font file inside the archive
    - a checksum of the font file
    - some font type information
* data about the model
    - the name of the model file inside the archive
    - a checksum of the model file
* data about the used image
    - font size
    - image size
    - starting location of the text

```json
{
    "version": 2,
    "build": 1533389537,
    "font": {
        "filename": "font.ttf",
        "checksum": "1171028651a0217165684f983cdf3a3b",
        "type": ".ttf",
        "name": "Arial",
        "family": "Arial"
    },
    "model": {
        "filename": "model.h5",
        "checksum": "1f9e8174aea018fb7a9f3f836fc688f9"
    },
    "image": {
        "font_size": 10,
        "image_size": [150, 12],
        "text_location": [0, 0]
    }
}
```

and the needed files (model and font).

```
$ python cli.py archive
Usage: cli.py archive [OPTIONS] COMMAND [ARGS]...

  Manage archives

Options:
  --help  Show this message and exit.

Commands:
  pack    Create an archive from the given files
  unpack  Extract the given archive
```


### Pack

```
$ python cli.py archive pack --help
Usage: cli.py archive pack [OPTIONS] FONT MODEL [ARCHIVE]

  Create an v2 archive from the given files

Options:
  --font-size INTEGER RANGE       The font size  [default: 10]
  --image-size <INTEGER INTEGER>...
                                  The size of the image  [default: 150, 12]
  --text-location <INTEGER INTEGER>...
                                  The starting location of the text  [default:
                                  0, 0]
  --help                          Show this message and exit.
```

This command packs the given files and metadata into an archive with version 2.

```
$ python cli.py archive pack Arial.ttf model.h5 model.zip
```


### Unpack

```
$ python cli.py archive unpack --help
Usage: cli.py archive unpack [OPTIONS] ARCHIVE [OUTPUT]

  Extract the given archive

Options:
  --help  Show this message and exit.
```

This command unpacks the given archive to the given directory or the current working directory if no output directory is given.

```
$ python cli.py archive unpack model.zip path/to/unpack/to/
$ ls path/to/unpack/to/
font.ttf meta.json model.h5
```
