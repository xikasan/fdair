# coding: utf-8

import numpy as np
import xtools as xt
import xtools.simulation as xs
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from fdair.models.model import FaultDetector
from fdair.utilities.dataloader import DataLoader, stack_10step


def main(conf):
    cf = conf if isinstance(conf, xt.Config) else xt.Config(conf)

    # model
    model = FaultDetector(cf.model.units, cf.model.size.input, cf.model.size.output)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=cf.train.lr)

    # data loader
    train_loader = build_loader(cf.data.train, cf.train.batch.size)
    test_loader  = build_loader(cf.data.test, cf.train.batch.size)

    # logger
    if cf.save.do:
        layers = "L{}_{}".format(*cf.model.units)
        cf.save.directory = xt.join(cf.save.directory, layers)
        cf.save.directory = xt.make_dirs_current_time(cf.save.directory)
        writer = tf.summary.create_file_writer(cf.save.directory)
        writer.set_as_default()
        cf.dump(xt.join(cf.save.directory, "config.yaml"))
        checkpoint = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(
            checkpoint,
            xt.join(cf.save.directory, "model"),
            max_to_keep=cf.save.model.num
        )
    logger = xs.ReplayBuffer({
        "epoch": 1,
        "loss": 1,
        "accuracy": 1
    }, capacity=cf.train.epoch.max)

    epoch = tf.Variable(0, dtype=tf.int64)
    step = tf.Variable(0, dtype=tf.int64)

    while True:
        epoch = epoch + 1

        for batch in train_loader:
            step = step + 1
            if step == 1:
                tf.summary.trace_on()

            loss = train(batch, model, optimizer)
            tf.summary.scalar("train/loss", tf.constant(loss), step=step)

            if step == 1:
                tf.summary.trace_export(
                    "FaultDetector",
                    step=epoch,
                    profiler_outdir=None
                )

        if cf.save.do and (epoch % cf.save.model.interval) == 0:
            manager.save()
            writer.flush()

        loss, acc = test(test_loader, model)
        tf.summary.scalar("test/loss",     tf.Variable(loss), step=epoch)
        tf.summary.scalar("test/accuracy", tf.Variable(acc),  step=epoch)
        logger.add(epoch=epoch.numpy(), loss=loss, accuracy=acc)
        print("epoch: {: 4.0f}\t loss: {:10.6f}\t accuracy: {:10.6f}".format(
            epoch.numpy(), loss, acc
        ))

        if epoch == cf.train.epoch.max:
            break

    # post train processing
    ret = xs.Retriever(logger.buffer())
    result = pd.DataFrame({
        "epoch": ret("epoch"),
        "loss": ret("loss"),
        "accuracy": ret("accuracy"),
    })

    result_name = xt.join(cf.save.directory, "result.csv")
    result.to_csv(result_name, index=False)
    print(result)
    xt.info("result saved", result_name)


def train(batch, model, optimizer):
    xs = batch.xs
    ys = np.squeeze(batch.ys).astype(np.int32)
    loss = train_body(xs, ys, model, optimizer)
    return loss.numpy()


@tf.function
def train_body(xs, ys, model, optimizer):
    with tf.GradientTape() as tape:
        ps = model(xs)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(ys, ps)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def test(loader, model):
    sum_loss = 0
    sum_acc  = 0
    for batch in loader:
        xs = batch.xs
        ys = np.squeeze(batch.ys).astype(np.int32)
        loss, acc = test_body(xs, ys, model)
        sum_loss += loss.numpy() * batch.size
        sum_acc  += acc.numpy()  * batch.size
    return sum_loss / loader.size(), sum_acc / loader.size()


@tf.function
def test_body(xs, ys, model):
    # predict and loss
    ps = model(xs)
    print(ps.shape, ys.shape)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(ys, ps)
    loss = tf.reduce_mean(loss)

    # accuracy
    acc = tf.argmax(ps, axis=1)
    acc = tf.cast(acc, tf.int32)
    acc = tf.abs(acc - ys)
    acc = 1 - tf.math.sign(acc)
    acc = tf.cast(acc, tf.float32)
    acc = tf.reduce_mean(acc)

    return loss, acc


def build_loader(data_list_file, batch_size):
    data_lists = np.loadtxt(data_list_file, dtype=str)
    loader = DataLoader(batch_size)
    loader.load_files(data_lists, pre_process=stack_10step)
    return loader


if __name__ == '__main__':
    xt.go_to_root()
    config = "experiments/2.train/train.yaml"
    main(config)
