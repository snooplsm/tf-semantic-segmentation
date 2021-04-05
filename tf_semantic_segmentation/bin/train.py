from os.path import basename
from tf_semantic_segmentation.models import get_model_by_name, models_by_name
from tf_semantic_segmentation.datasets import get_dataset_by_name, DataType, datasets_by_name, get_cache_dir, google_drive_records_by_tag, \
    download_records, DirectoryDataset, TFWriter, TFReader
from tf_semantic_segmentation.datasets.utils import convert2tfdataset
from tf_semantic_segmentation.losses import get_loss_by_name, losses_by_name
from tf_semantic_segmentation.metrics import metrics_by_name, get_metric_by_name, iou_score
from tf_semantic_segmentation.processing import dataset as preprocessing_ds
from tf_semantic_segmentation.processing import ColorMode
from tf_semantic_segmentation.settings import logger
from tf_semantic_segmentation.optimizers import get_optimizer_by_name, names as optimizer_choices
from tf_semantic_segmentation.utils import get_now_timestamp, kill_start_tensorboard, get_gpu_stats
from tf_semantic_segmentation import callbacks as custom_callbacks

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras import callbacks as kcallbacks

import os
import argparse
import shutil
import ast
import inspect
import types
import logging
import copy
import tempfile


def find_optimal_batch_size(args, batch_sizes=[pow(2, i) for i in range(16)], steps_per_epoch=-1):

    # reset loglevel to reduce printing
    current_level = logger.level
    logger.setLevel(logging.CRITICAL)

    current_bs = 0
    _args = copy.deepcopy(args)

    # only test with steps and buffer at bare min
    _args.validation_steps = 1
    _args.steps_per_epoch = steps_per_epoch
    _args.epochs = 1
    _args.buffer_size = 10
    _args.val_buffer_size = 1

    for batch_size in batch_sizes:
        print("testing batch size %d on model %s" % (batch_size, _args.model))
        try:
            # try to train for 30 seconds
            _args.batch_size = batch_size
            _args.log_dir = tempfile.mkdtemp()
            _ = train_test_model(_args)

            current_bs = batch_size
            shutil.rmtree(_args.log_dir)
        except tf.errors.ResourceExhaustedError as re:
            shutil.rmtree(_args.log_dir)
            break
        except tf.errors.InternalError as ie:
            shutil.rmtree(_args.log_dir)
            break

    logger.setLevel(current_level)
    return current_bs


def get_logdir_name(args):
    if args.run_name != None:
        name = args.run_name + "-" + get_now_timestamp()
    else:
        dataset = "record-%s" % os.path.dirname(args.record_dir) if args.record_dir else str(args.dataset)
        prefix = '%s-%s-bs%d-e%d-lr%.4f-%s-%s-%s' % (
            dataset, str(args.model), args.batch_size, args.epochs, args.learning_rate, args.optimizer, args.loss, args.final_activation
        )
        name = prefix + "-" + get_now_timestamp()

    return name


def get_args(args=None):

    color_modes = [int(cm) for cm in ColorMode]

    def str_list_type(x): return list(map(str, x.split(",")))

    def dict_type(x):
        if type(x) == dict:
            return x

        return ast.literal_eval(x)

    def float_list_type(x): return list(map(float, x.split(",")))
    def int_list_type(x): return [] if x.strip() == "" else list(map(int, x.split(",")))
    def tuple_type(x): return tuple(list(map(int, x.split(","))))

    def any_of(args):
        def any_of_type(x):
            x_list = list(map(str, x.split(",")))
            for t in x_list:
                assert(t in args), "%s is not in list %s" % (t, str(args))

            return x_list

        return any_of_type

    def any_enum(enum):
        def any_of_enum(x):
            x_list = list(map(str, x.split(",")))
            args = []
            for t in x_list:
                args.append(enum(t))

            return args

        return any_of_enum

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', '--size', default=None, type=tuple_type, help='size of the input images (height, width), inputs will be resized to given size')
    parser.add_argument('-m', '--model', default='erfnet', choices=list(models_by_name.keys()), help='model to train')
    parser.add_argument('-gpu', '--gpus', default=[0], type=int_list_type, help='gpus indexes to train on, e.g. train on 1 and 2: --gpus="1,2"')
    parser.add_argument('-cm', '--color_mode', default=0, type=int, choices=color_modes, help='Color mode: 0 (RGB), 1 (GRAY), 2 (NONE)')
    parser.add_argument('-o', '--optimizer', default='adam', choices=optimizer_choices, help='optimizer')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-l', '--loss', default='categorical_crossentropy', type=str, choices=list(losses_by_name.keys()), help='loss')
    parser.add_argument('-lm', '--metrics', default=['iou_score', 'f1_score', 'categorical_accuracy'],
                        type=any_of(metrics_by_name.keys()),
                        help='metrics, choices: %s' % (list(metrics_by_name.keys())))
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('-logdir', '--logdir', default=None, help='log dir (where the tensorboard log files and saved models go)')

    parser.add_argument('-delete', '--delete_logdir', action='store_true', help='if logdir exist and --delete_logdir, delete everything in it')
    parser.add_argument('-no_eval', '--no_evaluate', action='store_true', help='evaluates after training completes on the validation set')

    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('-ti', '--time_it', default=False, action='store_true', help='adds timing callback to measure images/sec and sec/batch')
    parser.add_argument('-ti_freq', '--time_it_log_freq', default=50, type=int, help='(info) log after x batches')
    parser.add_argument('-bufsize', '--buffer_size', default=50, type=int, help='number of examples to prefetch for training')
    parser.add_argument('-valbufsize', '--val_buffer_size', default=25, type=int, help='number of examples to prefetch for validation')
    parser.add_argument('-valfreq', '--validation_freq', default=1, type=int, help='validate every x epochs')
    parser.add_argument('-log', '--log_level', default='INFO', type=str, choices=['DEBUG', "NOTSET", "INFO", "WARN", "ERROR", "CRITICAL"], help='log level during training')
    parser.add_argument('-rm', '--resize_method', default='resize', type=str, choices=['resize', 'resize_with_pad', 'resize_with_crop_or_pad'], help='image resize method (when --size is specified)')
    parser.add_argument('-args', '--model_args', default={}, type=dict_type, help='arguments to supply to the model, e.g. unet: {"downsampling_method": "conv"}')
    parser.add_argument('--tpu_strategy', action='store_true', help='use the tpu strategy for training on tpus')
    parser.add_argument('--mixed_float16', action='store_true', help='use tf 2.1 feature to train a whole keras model on float16, REQUIRES TF 2.1')

    # run name
    parser.add_argument('-name', '--run_name', default=None, type=str, help='name of the run')

    # wandb
    parser.add_argument('-p', '--wandb_project', default=None, help='project name, if None wandb wont be used; uses `run_name` for nameing the current run')

    # weights
    parser.add_argument('-weights', '--model_weights', default=None, type=str, help='path to the model weights')
    parser.add_argument('-smv', '--saved_model_version', default=0, type=int, help='saved model version number')
    parser.add_argument('--no_save_model_weights', action="store_true", help='if specifed, do not save model weights')
    parser.add_argument('--no_export_saved_model', action="store_true", help='if specifed, do not export saved model (for tensorflow model server)')

    # data
    parser.add_argument('-data_dir', '--data_dir', default='/hdd/datasets', help='data directory')
    parser.add_argument('-rd', '--record_dir', default=None, help='if none, will be auto detected')
    parser.add_argument('-ro', '--record_options', default='GZIP', help='record compression options')
    parser.add_argument('-ds', '--dataset', default=None, choices=list(datasets_by_name.keys()), help='dataset to train on')
    parser.add_argument('-ds_args', '--dataset_args', default={}, type=dict_type, help='args for the dataset to initialize')
    parser.add_argument('-rtag', '--record_tag', default=None, choices=list(google_drive_records_by_tag.keys()), help='record tag for auto downloading records')
    parser.add_argument('-dir', '--directory', default=None, help='training model on a directory containing images and masks')
    parser.add_argument('-aug', '--augmentations', default=[], type=any_of(preprocessing_ds.augmentation_methods),
                        help='a list of augmentations (%s) separated by comma' % str(preprocessing_ds.augmentation_methods))

    parser.add_argument('-tog', '--train_on_generator', action='store_true', help='training using the tf.data.Dataset.from_generator for faster iterations and not creating a tfrecord')
    parser.add_argument('-steps', '--steps_per_epoch', default=-1, type=int, help='if not set, will be calculated based on the number of examples in the record')
    parser.add_argument('-valsteps', '--validation_steps', default=-1, type=int, help='if not set, will be calculated based on the number of examples in the record')

    # model
    parser.add_argument("-fa", '--final_activation', default='softmax', type=str, choices=['softmax', 'sigmoid'], help='activation of the final layer')
    parser.add_argument('-sum', '--summary', action='store_true', help='summarize model graph')

    # ray tune
    # parser.add_argument('-no-ip-address', '--node-ip-address', default=None)
    # parser.add_argument('-redis-address', '--redis-address', default=None)
    # parser.add_argument('-config-list', '--config-list', default=None)
    # parser.add_argument('-temp-dir', '--temp-dir', default=None)
    # parser.add_argument('--use-pickle', '--use-pickle', action='store_true')
    # parser.add_argument('-node-manager-port', '--node-manager-port', default=None)
    # parser.add_argument('-object-store-name', '--object-store-name', default=None)
    # parser.add_argument('-raylet-name', '--raylet-name', default=None)

    # callbacks
    parser.add_argument('--no_terminate_on_nan', action='store_true', help='if specified, do not add callback for terminating when nans occur')

    # model checkpoints
    parser.add_argument('--no_model_checkpoint', action='store_true', help='if specified, do not add callback for model checkpointing')
    parser.add_argument('-mc-monitor', '--model_checkpoint_monitor', default='val_loss', type=str, help='monitor metric/loss when checkpointing')
    parser.add_argument('-mc-no-sbo', '--no_save_best_only', action='store_true', help='if specified, do not save the best weights only')

    # auto start tensorboard
    parser.add_argument('--start_tensorboard', action='store_true', help='if specified, auto start tensorboard')
    parser.add_argument('---tensorboard_port', type=int, default=6006, help='port on which to auto start tensorboard on')

    # tensorboard
    parser.add_argument('--no_tensorboard_metrics', action='store_true', help='dont show default metrics/losses in tensorboard')
    parser.add_argument('--tensorboard_val_images', action='store_true', help='show val images in tensorboard/wandb')
    parser.add_argument('--tensorboard_test_images', action='store_true', help='show test images in tensorboard/wandb')
    parser.add_argument('--tensorboard_train_images', action='store_true', help='show train images in tensorboard/wandb')

    parser.add_argument('--tensorboard_train_images_update_batch_freq', type=int, default=1000,
                        help='show train images every n batch images in tensorboard/wandb')

    parser.add_argument('-num_tb_imgs', '--num_tensorboard_images', type=int, default=3, help='number of images displayed in tensorboard')
    parser.add_argument('-tb_images_freq', '--tensorboard_images_freq', type=int, default=1, help='after every $ epoch, log images [only used for test/val]')
    parser.add_argument('-binary_thresh', '--binary_threshold', type=float, default=0.5, help='values above threshold are rounded to 1.0, below to 0.0')
    parser.add_argument('-tb_uf', '--tensorboard_update_freq', default='batch', type=str, choices=['batch', 'epoch'], help='update frequency [batch or epoch] of tensorboard')
    parser.add_argument("--save_val_images", action='store_true', help='saves val images to logdir/val/samples')
    parser.add_argument("--save_train_images", action='store_true', help='saves val images to logdir/train/samples')
    parser.add_argument("--save_test_images", action='store_true', help='saves val images to logdir/test/samples')
    parser.add_argument("-v", "--visualizations", type=any_enum, default=custom_callbacks.DEFAULT_VISUALIZATIONS)
    parser.add_argument("--tb_draw_contours", action='store_true', help='draw contours in tensorboard')

    # early stopping
    parser.add_argument('--no_early_stopping', action='store_true', help='if specified, do not add callback for early stopping')
    parser.add_argument('-es_patience', '--early_stopping_patience', default=20, type=int, help='early stopping patience [epochs]')
    parser.add_argument('-es_mode', '--early_stopping_mode', default='min', type=str, help='early stopping mode')
    parser.add_argument('-es_monitor', '--early_stopping_monitor', default='val_loss', type=str, help='early stopping monitor metric/loss')
    parser.add_argument('-es_min_delta', '--early_stopping_min_delta', default=0, type=float,
                        help='early stopping monitor mindelta that counts for an improvement')

    # lr finder
    parser.add_argument('--find_lr', action='store_true', help='if specified, the learning rate finder runs')
    parser.add_argument('-fminlr', "--find_lr_min_lr", default=1e-9, type=float, help='minimum lr used to find the learning rate')
    parser.add_argument('-fmaxlr', "--find_lr_max_lr", default=1e-1, type=float, help='maxium lr used to find the learning rate')
    parser.add_argument('-fbeta', "--find_lr_beta", default=1e-1, type=float, help='beta used for changing the lr during lr find')
    parser.add_argument('-fstop', "--find_lr_stop_factor", default=4.0, type=float, help='beta used for changing the lr during lr find')

    # reduce lr on plateau
    parser.add_argument('--reduce_lr_on_plateau', action='store_true', help='if specified, do not add callback for reducing lr on plateau')
    parser.add_argument('-lr_patience', '--reduce_lr_patience', default=10, type=int, help='reduce lr on plateau patience in [epochs]')
    parser.add_argument('-lr_mode', '--reduce_lr_mode', default='min', type=str, help='reduce lr mode')
    parser.add_argument('-lr_factor', '--reduce_lr_factor', default=0.1, type=float, help='reduce lr factor')
    parser.add_argument('-lr_min_lr', '--reduce_lr_min_lr', default=1e-7, type=float, help='minimum learning rate when using reduce_lr_on_plateau')
    parser.add_argument('-lr_min_delta', '--reduce_lr_min_delta', default=0.0001, type=float, help='reduce lr min delta')
    parser.add_argument('-lr_monitor', '--reduce_lr_monitor', default='val_loss', type=str, help='reduce lr monitor')

    # mlflow
    parser.add_argument('-flow', '--mlflow', action='store_true', help='enable mlflow auto logging')
    parser.add_argument('-flow_exp', '--mlflow_experiment', help='mlflow experiment name, will create the experiment if it does not exist and ignored if mlflow_experiment_id is specified')
    parser.add_argument("-flow_exp_id", "--mlflow_experiment_id", help='id to the mlflow experiment', default=None)
    parser.add_argument('-flow_trackuri', '--mlflow_tracking_uri', help='tracking uri to the mlflow client', default=None)
    parser.add_argument('-flow_reguri', '--mlflow_registry_uri', help='registry uri to the mlflow client', default=None)
    parser.add_argument("--no_flow_log_images", action='store_true', help='do not log images when tensorboard is enabled')

    # notifications
    parser.add_argument('--notify', action='store_true', help='activate notifications')
    parser.add_argument('--slack_token', default=None, help='slack token')
    parser.add_argument('--slack_username', default="TFSemSeg", help='slack username that posts notifications')
    parser.add_argument('--slack_channel', default="training", help='slack channel that notifications will be posted to')

    args = parser.parse_args(args=args)

    # tf.get_logger().setLevel(args.log_level)
    logger.setLevel(args.log_level)

    return args


def setup_devices():
    try:
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as re:
        logger.warning("%s" % str(re))


def train_test_model(args, hparams=None, reporter=None):

    logger.info("setting up devices")
    # allow growth to precent memory errors
    setup_devices()

    logger.info("using tf version %s" % tf.__version__)

    logger.info("setting up callbacks")
    callbacks = args.callbacks if hasattr(args, 'callbacks') else []

    # setting up wandb
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config=args, name=args.run_name, sync_tensorboard=True, reinit=True)
        callbacks.append(wandb.keras.WandbCallback())

        if args.logdir is None:
            args.logdir = os.path.join("logs", args.wandb_project, "%s-%s" % (get_now_timestamp(), str(wandb_run.id)))
            logger.info("Using logdir %s, because None was specified" % args.logdir)

    if args.logdir is None:

        name = get_logdir_name(args)
        args.logdir = os.path.join("logs", "default", name)

    logger.info("logdir: %s" % args.logdir)
    if args.delete_logdir and os.path.isdir(args.logdir):
        logger.warning("delting everything in logdir %s" % args.logdir)
        shutil.rmtree(args.logdir)

    os.makedirs(args.logdir, exist_ok=True)

    # write hyperparameters as text summary
    with tf.summary.create_file_writer(os.path.join(args.logdir, 'train')).as_default():
        hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in vars(args).items()]
        tf.summary.text('hyperparameters', tf.stack(hyperparameters), step=0)

    if not args.no_tensorboard_metrics:
        callbacks.append(kcallbacks.TensorBoard(log_dir=args.logdir, histogram_freq=0, write_graph=True, profile_batch=0,
                                                write_images=False, write_grads=True, update_freq=args.tensorboard_update_freq))

    if not args.no_terminate_on_nan:
        callbacks.append(kcallbacks.TerminateOnNaN())

    if not args.no_model_checkpoint:
        callbacks.append(kcallbacks.ModelCheckpoint(os.path.join(args.logdir, "model-best.h5"),
                                                    monitor=args.model_checkpoint_monitor,  # val_loss default
                                                    verbose=1,
                                                    save_best_only=not args.no_save_best_only,
                                                    period=1))

    if not args.no_early_stopping:
        callbacks.append(kcallbacks.EarlyStopping(monitor=args.early_stopping_monitor,  # default: val_loss
                                                  mode=args.early_stopping_mode,  # default: min
                                                  min_delta=args.early_stopping_min_delta,
                                                  patience=args.early_stopping_patience,  # default: 20
                                                  verbose=1))

    if args.reduce_lr_on_plateau:
        callbacks.append(kcallbacks.ReduceLROnPlateau(monitor=args.reduce_lr_monitor, factor=args.reduce_lr_factor,
                                                      patience=args.reduce_lr_patience, min_lr=args.reduce_lr_min_lr, verbose=1,
                                                      mode=args.reduce_lr_mode, min_delta=args.reduce_lr_min_delta))

    if args.notify:
        callbacks.append(custom_callbacks.NotificationCallback(run_name=args.run_name,
                                                               token=args.slack_token,
                                                               username=args.slack_username,
                                                               channel=args.slack_channel))

    if hparams:
        from tensorboard.plugins.hparams import api as hp
        callbacks.append(hp.KerasCallback(args.logdir, hparams))

    if reporter:
        from ray.tune.integration.keras import TuneReporterCallback
        callbacks.append(TuneReporterCallback(reporter))

    if args.tpu_strategy:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    elif len(args.gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(args.gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:%d" % args.gpus[0])
    else:
        strategy = tf.distribute.MirroredStrategy(devices=['/gpu:%d' % gpu for gpu in args.gpus])

    global_batch_size = args.batch_size * (len(args.gpus) if len(args.gpus) > 0 else 1)

    assert(args.record_dir is not None or args.dataset is not None or args.record_tag is not None or args.directory is not None)

    logger.info("setting up dataset")
    ds = None  # will be used when in dataset mode

    if args.dataset or args.directory:
        if args.dataset and type(args.dataset) == str:
            cache_dir = get_cache_dir(args.data_dir, args.dataset)
            ds = get_dataset_by_name(args.dataset, cache_dir, args.dataset_args)
        elif args.dataset:
            ds = args.dataset
            cache_dir = get_cache_dir(args.data_dir, args.dataset.__class__.__name__)
        else:
            ds = DirectoryDataset(args.directory)
            cache_dir = args.directory

        assert(ds.num_classes > 0), "The dataset must have at least 1 class"
        logger.info("using dataset %s with %d classes" % (ds.__class__.__name__, ds.num_classes))

        if not args.train_on_generator:

            logger.info("writing records")

            record_dir = os.path.join(cache_dir, 'records')
            logger.info("using record dir %s" % record_dir)

            writer = TFWriter(record_dir, options=args.record_options)
            writer.write(ds)
            writer.validate(ds)
        else:
            record_dir = None

        num_classes = ds.num_classes
    elif args.record_dir:
        if not os.path.exists(args.record_dir):
            raise Exception("cannot find record dir %s" % args.record_dir)

        record_dir = args.record_dir
        num_classes = TFReader(record_dir, options=args.record_options).num_classes
    elif args.record_tag:
        record_tag = args.record_tag
        record_dir = os.path.join(args.data_dir, 'downloaded', record_tag)
        download_records(record_tag, record_dir)
        num_classes = TFReader(record_dir, options=args.record_options).num_classes
    else:
        raise Exception("cannot find either dataset/directory/record_dir or record_tag")

    if args.size and args.color_mode != ColorMode.NONE:
        input_shape = (args.size[0], args.size[1], 3 if args.color_mode == ColorMode.RGB else 1)

    elif args.train_on_generator:
        raise Exception("please specify the 'size' and 'color_mode' argument when training using the generator")
    else:
        if record_dir is None:
            raise Exception("record_dir cannot be None when trying to read record files")

        input_shape = TFReader(record_dir, options=args.record_options).input_shape
        input_shape = (input_shape[0], input_shape[1], 3 if args.color_mode == ColorMode.RGB else 1)

    logger.info("input shape: %s" % str(input_shape))

    try:
        if args.mixed_float16:
            logger.info("using mixed float16 precision, tf version >= 2.1 required")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        else:
            tf.keras.mixed_precision.set_global_policy(None)

    except Exception as e:
        logger.error("cannot set mixed precision policy, exception: %s" % str(e))

    # set scale mask based on sigmoid activation
    scale_mask = args.final_activation == 'sigmoid'

    if num_classes != 2 and args.final_activation == 'sigmoid':
        logger.error('do not choose sigmoid as the final activation when the dataset has more than 2 classes')
        return {}

    if args.final_activation == 'sigmoid':
        logger.warning('using only 1 output channel for sigmoid activation function to work')
        num_classes = 1

    logger.info('strategy: %s' % str(strategy))

    # check valid model args
    if args.model in models_by_name:
        valid_model_args = list(inspect.signature(models_by_name[args.model]).parameters.keys())

        for key in args.model_args.keys():
            if key not in valid_model_args:
                raise Exception("invalid model args; cannot find key %s in %s for model of name %s" % (key, str(valid_model_args), args.model))

    logger.info("creating model %s" % args.model)
    with strategy.scope():
        model_args = {'input_shape': input_shape, "num_classes": num_classes}
        model_args.update(args.model_args)

        if isinstance(args.model, str):
            model = get_model_by_name(args.model, model_args)
        elif isinstance(args.model, types.FunctionType):
            model = args.model(**model_args)
        else:
            logger.warning("using own model, please make sure num_classes and input_shape is correct")
            model = args.model

        if not args.no_save_model_weights:
            callbacks.append(custom_callbacks.SaveBestWeights(model, os.path.join(args.logdir, 'best-weights.h5')))

        if args.model_weights:
            logger.info("restoring model weights from %s" % args.model_weights)
            model.load_weights(args.model_weights)

        model = Model(model.input, Activation(args.final_activation, dtype='float32', name='predictions')(model.output))
        logger.info("output shape: %s" % model.output.shape)
        logger.info("input shape: %s" % model.input.shape)

        # loss and metrics
        loss = get_loss_by_name(args.loss)
        metrics = [get_metric_by_name(name) for name in args.metrics]

        logger.info("metrics: %s" % str(metrics))
        logger.info("loss: %s" % str(loss))

        opt = get_optimizer_by_name(args.optimizer, args.learning_rate)
        model.compile(optimizer=opt, loss=loss, metrics=metrics)  # metrics=losses

    if args.summary:
        model.summary()

    if args.train_on_generator:
        if ds is None:
            raise Exception("Dataset cannot be None when training with generator")
        train_ds = convert2tfdataset(ds, DataType.TRAIN)
        val_ds = convert2tfdataset(ds, DataType.VAL)
        reader = None  # no tfrecord reader
    else:
        logger.info("using tfreader to read record dir %s" % record_dir)
        reader = TFReader(record_dir, options=args.record_options)
        train_ds = reader.get_dataset(DataType.TRAIN)
        val_ds = reader.get_dataset(DataType.VAL)

    logger.info("building input pipeline")
    # train preprocessing
    train_preprocess_fn = preprocessing_ds.get_preprocess_fn(args.size, args.color_mode, args.resize_method, scale_mask=scale_mask)
    train_ds = train_ds.map(train_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if len(args.augmentations) == 0:
        augment_fn = None
    else:
        logger.info("applying augmentations %s" % str(args.augmentations))
        augment_fn = preprocessing_ds.get_augment_fn(args.size, global_batch_size, methods=args.augmentations)

    train_ds = preprocessing_ds.prepare_dataset(train_ds, global_batch_size, buffer_size=args.buffer_size, augment_fn=augment_fn)

    # val preprocessing
    val_preprocess_fn = preprocessing_ds.get_preprocess_fn(args.size, args.color_mode, args.resize_method, scale_mask=scale_mask)
    val_ds = val_ds.map(val_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = preprocessing_ds.prepare_dataset(val_ds, global_batch_size, buffer_size=args.val_buffer_size)

    # log images to tensorboard
    if (args.tensorboard_train_images and args.tensorboard_train_images_update_batch_freq > 0) or args.save_train_images:
        train_ds_images = convert2tfdataset(ds, DataType.TRAIN) if args.train_on_generator else reader.get_dataset(DataType.TRAIN)
        train_ds_images = train_ds_images.map(val_preprocess_fn, num_parallel_calls=1)
        train_ds_images = preprocessing_ds.prepare_dataset(train_ds_images, args.num_tensorboard_images, buffer_size=100, shuffle=True, prefetch=False)
        train_prediction_callback = custom_callbacks.BatchPredictionCallback(model, os.path.join(args.logdir, 'train'), train_ds_images,
                                                                             scaled_mask=scale_mask,
                                                                             binary_threshold=args.binary_threshold,
                                                                             save_images=args.save_train_images,
                                                                             mlflow_logging=not args.no_flow_log_images and args.mlflow,
                                                                             visualizations=args.visualizations,
                                                                             save_to_tensorboard=args.tensorboard_train_images,
                                                                             update_freq=args.tensorboard_train_images_update_batch_freq)
        callbacks.append(train_prediction_callback)
        train_prediction_callback.on_batch_end(-1, {})

    if args.tensorboard_val_images or args.save_val_images:
        val_ds_images = convert2tfdataset(ds, DataType.VAL) if args.train_on_generator else reader.get_dataset(DataType.VAL)
        val_ds_images = val_ds_images.map(val_preprocess_fn, num_parallel_calls=1)
        val_ds_images = preprocessing_ds.prepare_dataset(val_ds_images, args.num_tensorboard_images, buffer_size=1, shuffle=False, prefetch=False, take=args.num_tensorboard_images)
        val_prediction_callback = custom_callbacks.EpochPredictionCallback(model, os.path.join(args.logdir, 'validation'), val_ds_images,
                                                                           scaled_mask=scale_mask,
                                                                           binary_threshold=args.binary_threshold,
                                                                           save_images=args.save_val_images,
                                                                           mlflow_logging=not args.no_flow_log_images and args.mlflow,
                                                                           visualizations=args.visualizations,
                                                                           save_to_tensorboard=args.tensorboard_val_images,
                                                                           update_freq=args.tensorboard_images_freq)
        callbacks.append(val_prediction_callback)
        val_prediction_callback.on_epoch_end(-1, {})

    if args.tensorboard_test_images or args.save_test_images:
        test_ds_images = convert2tfdataset(ds, DataType.TEST) if args.train_on_generator else reader.get_dataset(DataType.TEST)
        test_ds_images = test_ds_images.map(val_preprocess_fn, num_parallel_calls=1)
        test_ds_images = preprocessing_ds.prepare_dataset(test_ds_images, args.num_tensorboard_images, buffer_size=1, shuffle=False, prefetch=False, take=args.num_tensorboard_images)
        test_prediction_callback = custom_callbacks.EpochPredictionCallback(model, os.path.join(args.logdir, 'test'), test_ds_images,
                                                                            scaled_mask=scale_mask,
                                                                            save_images=args.save_test_images,
                                                                            binary_threshold=args.binary_threshold,
                                                                            mlflow_logging=not args.no_flow_log_images and args.mlflow,
                                                                            visualizations=args.visualizations,
                                                                            save_to_tensorboard=args.tensorboard_test_images,
                                                                            update_freq=args.tensorboard_images_freq)
        callbacks.append(test_prediction_callback)
        test_prediction_callback.on_epoch_end(-1, {})

    if args.start_tensorboard:
        kill_start_tensorboard(args.logdir, port=args.tensorboard_port)

    if args.steps_per_epoch != -1:
        steps_per_epoch = args.steps_per_epoch
    elif args.train_on_generator:
        steps_per_epoch = ds.num_examples(DataType.TRAIN) // global_batch_size
    else:
        logger.warning("Reading total number of input samples, cause no steps were specifed. This may take a while.")
        steps_per_epoch = reader.num_examples(DataType.TRAIN) // global_batch_size

    # add timing callback after steps per epoch is known
    if args.time_it:
        callbacks.append(custom_callbacks.TimingCallback(args.batch_size, steps_per_epoch * global_batch_size, log_interval=args.time_it_log_freq))

    if args.find_lr:
        find_lr_logdir = os.path.join(args.logdir, 'lr-finder')
        callbacks.append(custom_callbacks.LRFinder(
            model, steps_per_epoch, find_lr_logdir, args.epochs, args.find_lr_min_lr, args.find_lr_max_lr, args.find_lr_stop_factor, args.find_lr_beta
        ))

    if args.validation_steps != -1:
        validation_steps = args.validation_steps
    elif args.train_on_generator:
        validation_steps = ds.num_examples(DataType.VAL) // global_batch_size
    else:
        logger.warning("Reading total number of input val samples, cause no val_steps were specifed. This may take a while.")
        validation_steps = reader.num_examples(DataType.VAL) // global_batch_size

    if args.mlflow:

        import mlflow
        from mlflow.exceptions import MlflowException

        if args.mlflow_experiment_id:
            experiment_id = args.mlflow_experiment_id

        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        if args.mlflow_registry_uri:
            mlflow.set_registry_uri(args.mlflow_registry_uri)

        elif args.mlflow_experiment:
            try:
                logger.debug(f"try creating mlflow experiment {args.mlflow_experiment}")
                experiment_id = mlflow.create_experiment(args.mlflow_experiment)
                experiment = mlflow.get_experiment(experiment_id)
                logger.debug(f"created lflow experiment with id {experiment_id}")

            except MlflowException:
                experiment = mlflow.get_experiment_by_name(args.mlflow_experiment)
                experiment_id = experiment.experiment_id
                logger.debug(f"using existing mlflow experiment with id={experiment_id}")
        else:
            experiment_id = None

        params = {k: str(v) for k, v in vars(args).items()}

        if experiment_id or args.run_name:
            run = mlflow.start_run(experiment_id=experiment_id, run_name=args.run_name)
            logger.info("mlflow run id=%s" % (run.info.run_id))
        mlflow.tensorflow.autolog()
        mlflow.log_params(params)
        try:
            mlflow.log_param('num_gpus', len(tf.config.list_physical_devices(device_type='GPU')))
            for key, value in get_gpu_stats().items():
                mlflow.log_param("gpu_stats." + key, value)

        except Exception as e:
            logger.error("could not log mlflow extra params %s" % str(e))
            pass

    # model fitting
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps,
                        callbacks=callbacks, epochs=args.epochs, validation_freq=args.validation_freq)

    if not args.no_evaluate:
        results = model.evaluate(val_ds, steps=validation_steps)
    else:
        results = None

    # saved model export
    saved_model_path = os.path.join(args.logdir, 'saved_model', str(args.saved_model_version))

    if os.path.exists(saved_model_path) and not args.no_export_saved_model:
        shutil.rmtree(saved_model_path)

    if not args.no_export_saved_model:
        logger.info("exporting saved model to %s" % saved_model_path)
        model.save(saved_model_path, save_format='tf')

    return {"evaluate": results, "history": history, 'callbacks': callbacks}, model


def main():
    args = get_args()
    results, _ = train_test_model(args)
    print("results: ", results)


if __name__ == "__main__":
    main()
