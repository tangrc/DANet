#!/usr/bin/env python
# -*- coding:utf8 -*-
'''
  @Datetime: 2021-11-20 15:11:38
  @Author:   shiliu
'''
import sys
import os

cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(cur_path, '..')))
sys.path.append(os.path.abspath(os.path.join(cur_path, '../..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from tensorflow.contrib.layers.python.layers.feature_column import _EmbeddingColumn, _RealValuedColumn

import global_var as gl
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from prada_model_ops.metrics import auc
from tensorflow.python.ops import metrics
from prada_interface.algorithm import Algorithm
from model_util.fg import FgParser
from model_util.util import *
from model_util.attention import attention as atten_func
from model_util.gate import gate_layer as gate_func
from optimizer.adagrad_decay import SearchAdagradDecay
from optimizer.adagrad import SearchAdagrad
from optimizer.gradient_decent import SearchGradientDecent
from optimizer.gradient import SearchGradient
from optimizer import optimizer_ops as myopt
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
from requests.exceptions import ConnectionError
from model_util.attention import feedforward

import numpy as np
from model_util import odps_io as myodps

optimizer_dict = {
    "AdagradDecay": lambda opt_conf, global_step: SearchAdagradDecay(opt_conf).get_optimizer(global_step),
    "Adagrad": lambda opt_conf, global_step: SearchAdagrad(opt_conf).get_optimizer(global_step),
    "GradientDecentDecay": lambda opt_conf, global_step: SearchGradientDecent(opt_conf).get_optimizer(global_step),
    "GradientDecent": lambda opt_conf, global_step: SearchGradient(opt_conf).get_optimizer(global_step)
}


class DANet(Algorithm):
    def init(self, context):
        self.context = context
        self.logger = self.context.get_logger()
        gl._init()
        gl.set_value('logger', self.logger)
        self.config = self.context.get_config()
        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']
            self.metric_conf = v['metrics']
        if self.model_name is None:
            self.model_name = "CVR"
        self.main_column_blocks = []
        self.bias_column_blocks = []
        self.dcm_context_column_blocks = []
        self.expert_name_list = []
        if self.algo_config.get('main_columns') is not None:
            arr_blocks = self.algo_config.get('main_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.main_column_blocks.append(block)
        else:
            raise RuntimeError("main_columns must be specified.")

        if self.algo_config.get('bias_columns') is not None:
            arr_blocks = self.algo_config.get('bias_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.bias_column_blocks.append(block)

        if self.algo_config.get('dcm_context_columns') is not None:
            arr_blocks = self.algo_config.get('dcm_context_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.dcm_context_column_blocks.append(block)
        else:
            raise RuntimeError("gate_columns must be specified.")

        self.seq_column_blocks = []
        self.seq_column_len = {}
        self.seq_column_atten = {}
        self.seq_column_value = {}
        self.seq_column_v = {}

        if self.algo_config.get('seq_column_blocks') is not None:
            arr_blocks = self.algo_config.get('seq_column_blocks').split(';', -1)
            for block in arr_blocks:
                if block == "":
                    continue
                arr = block.split(':', -1)
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                    if arr[0] == 'buy_seq_list':
                        self.seq_column_v['dcm_user_atten_v'] = self.model_conf["model_hyperparameter"].get('dcm_buy_seq_v_feats',
                                                                                                                 [])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]
                if len(arr[2]) > 0:
                    self.seq_column_atten[arr[0] + '_atten_item'] = arr[2]
                if len(arr[3]) > 0:
                    self.seq_column_value[arr[0] + '_atten_value'] = self.model_conf["model_hyperparameter"].get(arr[3],
                                                                                                                 [])
        self.atten_collections_dnn_hidden_layer = "{}_atten_dnn_hidden_layer".format(self.model_name)
        self.atten_collections_dnn_hidden_output = "{}_atten_dnn_hidden_output".format(self.model_name)
        self.main_collections_dnn_hidden_layer = "{}_main_dnn_hidden_layer".format(self.model_name)
        self.main_collections_dnn_hidden_output = "{}_main_dnn_hidden_output".format(self.model_name)
        self.bias_collections_dnn_hidden_layer = "{}_bias_dnn_hidden_layer".format(self.model_name)
        self.bias_collections_dnn_hidden_output = "{}_bias_dnn_hidden_output".format(self.model_name)
        self.gate_collections_dnn_hidden_layer = "{}_gate_dnn_hidden_layer".format(self.model_name)
        self.gate_collections_dnn_hidden_output = "{}_gate_dnn_hidden_output".format(self.model_name)
        self.layer_dict = {}
        self.sequence_layer_dict = {}
        self.metrics = {}
        self.sink = context.get_sink()
        self.fg = FgParser(self.config.get_fg_config())
        self.debug_tensor_collector = {}
        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")

    def variable_scope(self, *args, **kwargs):
        kwargs['partitioner'] = partitioned_variables.min_max_variable_partitioner(
            max_partitions=self.config.get_job_config("ps_num"),
            min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
        kwargs['reuse'] = tf.AUTO_REUSE
        return tf.variable_scope(*args, **kwargs)

    def inference(self, features, feature_columns):
        self.feature_columns = feature_columns
        self.features = features
        self.logger.info("[LogAllFeature] %s" % self.features)
        self.logger.info("[LogAllFeatureColumn] %s" % self.feature_columns)
        self.embedding_layer(features, feature_columns)
        self.sequence_layer()
        self.dpn_logits = self.dpn_net_backbone()
        self.logits = self.ipn_net_backbone()
        return self.logits

    def loss(self, logits, label):
        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            self.label = label
            self.logits = logits
            self.reg_loss_f()
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.label)
            mse_loss = tf.reduce_mean(tf.square(self.discount_label - self.dpn_logits))
            self.loss_op = tf.reduce_mean(loss) + self.model_conf['model_hyperparameter']['mse_loss_weight'] * mse_loss + self.reg_loss
            return self.loss_op

    def predictions(self, logits):
        with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
            self.prediction = tf.sigmoid(logits)
            return self.prediction

    def add_sample_trace_dict(self, key, value):
        try:
            self.sample_trace_dict[key] = tf.sparse_tensor_to_dense(value, default_value="")
        except:
            self.sample_trace_dict[key] = value

    def optimizer(self, context, loss_op):
        '''
        return train_op
        '''
        with tf.variable_scope(
                name_or_scope="Optimize",
                partitioner=partitioned_variables.min_max_variable_partitioner(
                    max_partitions=self.config.get_job_config("ps_num"),
                    min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                ),
                reuse=tf.AUTO_REUSE):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = self.update_op(name=self.model_name)

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = self.get_optimizer(opt_name, opt_conf, self.global_step)
                if 'scope' not in opt_conf or opt_conf["scope"] == "Global":
                    global_opt_name = opt_name
                    global_optimizer = optimizer
                else:
                    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=opt_conf["scope"])
                    if len(vars) != 0:
                        for var in vars:
                            if var in global_opt_vars:
                                global_opt_vars.remove(var)
                        train_op, _, _ = myopt.optimize_loss(
                            loss=loss_op,
                            global_step=self.global_step,
                            learning_rate=opt_conf.get("learning_rate", 0.01),
                            optimizer=optimizer,
                            # update_ops=update_ops,
                            clip_gradients=opt_conf.get('clip_gradients', 5),
                            variables=vars,
                            increment_global_step=False,
                            summaries=myopt.OPTIMIZER_SUMMARIES)
                        train_ops.append(train_op)
            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=loss_op,
                    global_step=self.global_step,
                    learning_rate=self.opts_conf[global_opt_name].get("learning_rate", 0.01),
                    optimizer=global_optimizer,
                    # update_ops=update_ops,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_ops = state_ops.assign_add(self.global_step, 1).op

    def update_op(self, name):
        update_ops = []
        start = ('Share') if name is None else ('Share', name)
        for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            if update_op.name.startswith(start):
                update_ops.append(update_op)
        return update_ops

    def get_optimizer(self, opt_name, opt_conf, global_step):
        optimizer = None
        for name in optimizer_dict:
            if opt_name == name and isinstance(optimizer_dict[name], str):
                optimizer = optimizer_dict[name]
                break
            elif opt_name == name:
                optimizer = optimizer_dict[name](opt_conf, global_step)
                break

        return optimizer

    def build_graph(self, context, features, feature_columns, labels):
        self.set_global_step()
        self.inference(features, feature_columns)
        self.loss(self.logits, labels)
        self.optimizer(context, self.loss_op)
        self.predictions(self.logits)


    def set_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def set_reset_op(self):
        self.reset_auc_ops, self.localvar = self.reset_variables(collection_key=tf.GraphKeys.LOCAL_VARIABLES,
                                                                 matchname='%s_Metrics/%s-auc' % (
                                                                     self.model_name, self.model_name))
        # matchname的传参是用的summary的scope，(tp,fn,tn,fp)

    def reset_variables(self, collection_key=tf.GraphKeys.LOCAL_VARIABLES, matchname='auc/', not_match=None):
        localv = tf.get_collection(collection_key)
        localv = [x for x in localv if matchname in x.name]
        if not_match is not None:
            localv = [x for x in localv if not_match not in x.name]
        retvops = [tf.assign(x, array_ops.zeros(shape=x.get_shape(), dtype=x.dtype)) for x in localv]
        if len(retvops) == 0:
            return None, None
        retvops = tf.tuple(retvops)
        return retvops, localv

    def embedding_layer(self, features, feature_columns):
        with tf.variable_scope(name_or_scope="Embedding_Layer",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for block_name in (self.main_column_blocks
                               + self.bias_column_blocks
                               + self.seq_column_len.values()
                               + self.gate_column_blocks):
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                self.logger.info("block_name:%s, len(feature_columns[block_name])=%d" %
                                 (block_name, len(feature_columns[block_name])))

                self.layer_dict[block_name] = layers.input_from_feature_columns(features,
                                                                                feature_columns=feature_columns[
                                                                                    block_name],
                                                                                scope=scope)

        self.sequence_layer_dict = self.build_sequence(self.seq_column_blocks, self.seq_column_len, "seq")
        self.source = tf.cast(self.features.get("source", None),dtype=tf.float32)
        self.discount_label =tf.cast(self.features.get("discount_zk_label", None),dtype=tf.float32)
        self.discount_rate_times = tf.cast(self.features.get("discount_rates_time", None), dtype=tf.string)
        with tf.variable_scope(name_or_scope="atten_input_from_feature_columns",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for atten_block_name in (self.seq_column_atten.values()):
                if len(atten_block_name) <= 0: continue
                if atten_block_name not in feature_columns or len(feature_columns[atten_block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for atten" % atten_block_name)
                self.layer_dict[atten_block_name] = layers.input_from_feature_columns(features,
                                                                                      feature_columns[atten_block_name],
                                                                                      scope=scope)

    def sequence_layer(self):
        for block_name in self.seq_column_blocks:
            attention_layer = [self.layer_dict[self.seq_column_atten[block_name + '_atten_item']]]
            attention = tf.concat(attention_layer, axis=1)
            self.logger.info("Debug!! attention of {} is {}".format(block_name, attention))
            target_atten = self.build_query_attention(
                self.sequence_layer_dict,
                self.seq_column_len,
                attention,
                None,
                block_name,
                self.model_conf['model_hyperparameter']['atten_param']['ma_num_units'],
                self.model_conf['model_hyperparameter']['atten_param']['ma_num_output_units'],
                self.model_conf['model_hyperparameter']['atten_param']['num_heads']
            )
            self.layer_dict[block_name] = target_atten


    def concat_features(self, inputs):
        return tf.concat(values=inputs, axis=1)

    def ipn_net_input(self):
        ipn_net_layer = []
        for block_name in (self.main_column_blocks + self.seq_column_blocks):
            if not self.layer_dict.has_key(block_name):
                raise ValueError('[Ipn net, layer dict] does not has block : {}'.format(block_name))
            ipn_net_layer.append(self.layer_dict[block_name])
            self.logger.info('[ipn_net_layer] add %s' % block_name)

        ipn_net_input = tf.concat(values=ipn_net_layer, axis=1)
        return ipn_net_input

    def dcm_content_net_input(self):
        dcm_context_net_layer = []
        for block_name in self.dcm_context_column_blocks:
            if not self.layer_dict.has_key(block_name):
                raise ValueError('[Dcm net, layer dict] does not has block : {}'.format(block_name))
            dcm_context_net_layer.append(self.layer_dict[block_name])
            self.logger.info('[dcm_context_net_layer] add %s' % block_name)

        dcm_context_net_input = tf.concat(values=dcm_context_net_layer, axis=1)
        return dcm_context_net_input

    def dcm_user_net_input(self):
        attention_layer = [self.layer_dict[self.seq_column_atten['buy_seq_list_atten_item']]]
        attention = tf.concat(attention_layer, axis=1)
        self.logger.info("Debug!! attention of {} is {}".format("buy_seq_list", attention))
        dcm_split_buy_atten = self.build_query_attention(
            self.sequence_layer_dict,
            self.seq_column_len,
            attention,
            None,
            "buy_seq_list",
            self.model_conf['model_hyperparameter']['atten_param']['ma_num_units'],
            self.model_conf['model_hyperparameter']['atten_param']['ma_num_output_units'],
            self.model_conf['model_hyperparameter']['atten_param']['num_heads'],
            is_dcm = True
        )
        self.layer_dict["dcm_split_buy_seq"] = dcm_split_buy_atten
        return self.layer_dict["dcm_split_buy_seq"]


    def ipn_net_backbone(self):
        self.ipn_net_emb = self.ipn_net_input()
        self.ipn_net = self.mlp(self.ipn_net_emb, self.model_conf['model_hyperparameter']['dnn_hidden_units'],
                            "ipn_net")
        self.ipn_concat_emb = tf.concat([self.ipn_net, self.dpn_concat_emb], axis=1)
        self.ipn_net_output = self.mlp(self.ipn_concat_emb, self.model_conf['model_hyperparameter']['dnn_hidden_units'],
                                "ipn_net")
        self.ipn_logits = layers.linear(
            self.ipn_net_output,
            1,
            scope="ipn_net",
            variables_collections=[self.main_collections_dnn_hidden_layer],
            outputs_collections=[self.main_collections_dnn_hidden_output],
            biases_initializer=None)
        return self.ipn_logits


    def dpn_net_backbone(self):
        self.dcm_user_emb = self.dcm_user_net_input()
        self.dcm_context_emb = self.dcm_content_net_input()
        self.dcm_user_output = self.mlp(self.dcm_user_emb, self.model_conf['model_hyperparameter']['bias_dnn_hidden_units'],
                                "dcm_user_net")
        self.dcm_context_gate_output = self.mlp(self.dcm_context_emb, self.model_conf['model_hyperparameter']['gate_hidden_units'],
                                     "dcm_context_net")
        self.dcm_gate_mutiply_emb = tf.multiply(self.dcm_user_output,self.dcm_context_gate_output)
        self.low_freq_emb, self.high_freq_emb = self.tftm_net()
        self.dpn_concat_emb = tf.concat([self.dcm_gate_mutiply_emb, self.low_freq_emb,self.high_freq_emb], axis=1)
        self.dpn_net = self.mlp(self.dpn_concat_emb, self.model_conf['model_hyperparameter']['dnn_hidden_units'],
                                "dpn_net")
        self.debug_tensor_collector['dpn_net'] = self.dpn_net
        dpn_logits = layers.linear(
            self.dpn_net,
            1,
            scope="dpn_net",
            variables_collections=[self.main_collections_dnn_hidden_layer],
            outputs_collections=[self.main_collections_dnn_hidden_output],
            biases_initializer=None)
        return dpn_logits


    def tftm_net(self):
        flattened_tensor = tf.squeeze(self.discount_rate_times)
        split_tensor = tf.strings.split(flattened_tensor, sep=',')
        dense_split_tensor = tf.sparse.to_dense(split_tensor, default_value="1")
        dense_split_tensor = tf.reshape(dense_split_tensor, [-1, 97])
        dr_time_tensor = tf.cast(dense_split_tensor, dtype=tf.float32)
        fft_result = tf.fft(tf.cast(dr_time_tensor, tf.complex64))
        def filter_mlp(fft_result):
            hidden_layer = tf.layers.dense(tf.cast(fft_result,tf.float32), units=64, activation=tf.nn.relu)
            output_layer = tf.layers.dense(hidden_layer, units=97, activation=None)
            return output_layer
        low_freq_signal = filter_mlp(fft_result)
        high_freq_signal = fft_result - tf.cast(low_freq_signal,tf.complex64)
        low_freq_time_domain = tf.ifft(tf.cast(low_freq_signal, tf.complex64))
        high_freq_time_domain = tf.ifft(tf.cast(high_freq_signal, tf.complex64))
        return low_freq_time_domain, high_freq_time_domain

    def mlp(self, inputs, dnn_hidden_units, name):
        with self.variable_scope(
                name_or_scope="{}_Network_Part_{}".format(self.model_name, name),
                partitioner=partitioned_variables.min_max_variable_partitioner(
                    max_partitions=self.config.get_job_config("ps_num"),
                    min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
        ):
            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                    with self.variable_scope(
                            name_or_scope="hidden_layer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        net = layers.fully_connected(
                            inputs,
                            num_hidden_units,
                            getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.bias_collections_dnn_hidden_layer],
                            outputs_collections=[self.bias_collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                'batch_norm', True) else None,
                            normalizer_params={"scale": True, "is_training": self.is_training})

                if self.model_conf['model_hyperparameter']['need_dropout']:
                    net = tf.layers.dropout(
                        net,
                        rate=self.model_conf['model_hyperparameter']['dropout_rate'],
                        noise_shape=None,
                        seed=None,
                        training=self.is_training,
                        name=None)
        return net


    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_tmp = []
        for reg_loss in reg_losses:
            if reg_loss.name.startswith(self.model_name) or reg_loss.name.startswith('Share'):
                reg_tmp.append(reg_loss)
        self.reg_loss = tf.reduce_sum(reg_tmp)
        self.logger.info('regularization_variable: {}'.format(reg_tmp))

    def mark_output(self, prediction):
        with tf.name_scope("%s_Mark_Output" % self.model_name):
            logistic = tf.identity(prediction, name="rank_predict")

    def run_predict(self, context, mon_session, task_index, thread_index):
        if int(thread_index) != 0:  # predict with one thread
            self.logger.info("Skip thread_ind==%s" % str(thread_index))
            return

        self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                          local_mode=False, odps_user=self.algo_config.get('odps_user'))
        self.tablewriter = myodps.getTableWriter(self.odps,
                                                 self.algo_config.get('table_name'),
                                                 task_id=task_index,
                                                 ds_output='eval_part',
                                                 local_mode=False)

        predict_step = self.algo_config.get('predict_max_step', 100)
        localcnt = 0
        id_feature_tensor = self.features["id"]
        try:
            id_feature_tensor = tf.sparse_tensor_to_dense(id_feature_tensor, default_value="")
            self.logger.info("#qlLog# self.id_feature_tensor")
        except:
            pass

        run_ops = [self.prediction, self.label, id_feature_tensor]
        # self.debug_tensor_collector['share_price_level_emb'] = tf.get_default_graph().get_tensor_by_name("Embedding_Layer/price_level_catleaf_shared_embedding/price_level_catleaf")
        # self.debug_tensor_collector['share_price_level_emb'] = tf.get_default_graph().get_tensor_by_name("Embedding_Layer/price_level_catleaf_shared_embedding/price_level_catleaf/part_0:0")

        debug_tensor_names = []
        for tensor_name, tensor in self.debug_tensor_collector.items():
            debug_tensor_names.append(tensor_name)
            run_ops.append(tensor)

        print('global_variables')
        for variable_name in tf.global_variables():
            print(variable_name)

        while True:
            localcnt += 1
            feed_dict = {'training:0': False}

            try:
                run_res = mon_session.run(run_ops, feed_dict=feed_dict)
                prob, y, qid = run_res[:3]
                records = []
                for i in range(len(prob)):
                    one_res = [str(prob[i][0]), str(y[i][0]), str(qid[i][0])]
                    ex = []
                    for tensor_name, tensor_value in zip(debug_tensor_names, run_res[3:]):
                        ex.append('{}={}'.format(tensor_name, str(tensor_value[i].tolist())))
                    if len(ex) > 0:
                        ex_str = '#'.join(ex)
                        one_res.append(ex_str)
                    else:
                        one_res.append('-')
                    records.append(one_res)

                self.tablewriter.write(task_index, records)
                self.logger.info(
                    'model_name=%s, size=%s,  step_left=%s' % (self.model_name, str(len(prob)), str(predict_step)))

                predict_step -= 1
                if predict_step < 1:
                    break
            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                self.logger.info("Reset table writer")
                self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                                  local_mode=False, odps_user=self.algo_config.get('odps_user'))
                self.tablewriter = myodps.getTableWriter(self.odps,
                                                         self.algo_config.get('table_name'),
                                                         task_id=task_index,
                                                         ds_output='eval_part',
                                                         local_mode=False)
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

        try:
            if self.tablewriter is not None:
                self.tablewriter.close()
        except ConnectionError as e:
            self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            self.logger.info("Reset table writer when close")
            self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                              local_mode=False, odps_user=self.algo_config.get('odps_user'))
            self.tablewriter = myodps.getTableWriter(self.odps,
                                                     self.algo_config.get('table_name'),
                                                     task_id=task_index,
                                                     ds_output='eval_part',
                                                     local_mode=False)
            if int(task_index) != 0:
                raise RuntimeError("Reset table writer when close")

            time.sleep(60)
            self.logger.info("Finish run_predict, sleep")

    def run_train(self, context, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.global_step, self.loss_op, self.metrics, self.label, self.localvar]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    global_step, loss, metrics, labels, flocalv = mon_session.run(
                        run_ops, feed_dict=feed_dict)
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.train_ops)
                    global_step, loss, metrics, labels, flocalv, _ = mon_session.run(
                        run_ops, feed_dict=feed_dict)

                if len(self.localvar) > 0:
                    index = np.array([0, -1])
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([x.name for x in self.localvar]))
                    # localvar值为： CTR_Metrics / CTR - auc / true_positives:0 //
                    #              CTR_Metrics / CTR - auc / false_negatives:0 //
                    #              CTR_Metrics / CTR - auc / true_negatives:0 //
                    #              CTR_Metrics / CTR - auc / false_positives:0
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([str(x[index]) for x in flocalv]))

                auc, totalauc, decay_auc = metrics['scalar/auc'], metrics['scalar/total_auc'], metrics['scalar/acc_auc']
                self.logger.info(
                    'Global_Step:{}, poslabel:{}, loss={}, auc={}, totalauc={}, decay_auc={}, thread={}'.format(
                        str(global_step),
                        str(labels.sum()),
                        str(loss),
                        str(auc),
                        str(totalauc),
                        str(decay_auc),
                        str(thread_index)))

                newmark = np.max(flocalv[0][np.array([0, -1])])  # 是true_positives累积到达20000,则重置total auc
                if newmark > self.metric_conf['auc_compute'].get('true_positives', 20000):
                    self.logger.info("positive_num now:{}".format(str(newmark)))
                    self.logger.info("auc_reset_step:{}".format(str(1000)))
                    self.logger.info('reset auc ops run')
                    index = np.array([0, -1])
                    flocalv = mon_session.run(self.reset_auc_ops, feed_dict=feed_dict)
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([x.name for x in self.localvar]))
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([str(x[index]) for x in flocalv]))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

    def run_evaluate(self, context, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.global_step_add, self.global_step, self.metrics, self.label, self.localvar]
            try:
                feed_dict = {'training:0': False}
                _, global_step, metrics, labels, flocalv = mon_session.run(
                    run_ops, feed_dict=feed_dict)
                if len(self.localvar) > 0:
                    index = np.array([0, -1])
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([x.name for x in self.localvar]))
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([str(x[index]) for x in flocalv]))

                auc, totalauc = metrics['scalar/auc'], metrics['scalar/total_auc']
                self.logger.info(
                    'Global_Step:{}, poslabel:{}, auc={}, totalauc={} thread={}'.format(
                        str(global_step),
                        str(labels.sum()),
                        str(auc),
                        str(totalauc),
                        str(thread_index)))
                newmark = np.max(flocalv[0][np.array([0, -1])])
                if newmark > self.metric_conf['auc_compute'].get('true_positives', 20000):
                    self.logger.info("positive_num now:{}".format(str(newmark)))
                    self.logger.info("auc_reset_step:{}".format(str(1000)))
                    self.logger.info('reset auc ops run')
                    index = np.array([0, -1])
                    flocalv = mon_session.run(self.reset_auc_ops, feed_dict=feed_dict)
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([x.name for x in self.localvar]))
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([str(x[index]) for x in flocalv]))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))


    def build_query_attention(self, sequence_layer_dict, seq_column_len, attention_layer, attention_dict, block_name,
                              ma_num_units=64, ma_num_output_units=64, num_heads=2, is_dcm =False):
        logger.info("Debug!! Start Build Query Attention: {}".format(block_name))
        # layer_dict = {}
        # weight_dict = {}
        if sequence_layer_dict is None or block_name not in sequence_layer_dict.keys():
            logger.info(
                "Debug!! Build Query Attention error. name is {}, dict is {}".format(block_name, sequence_layer_dict))
            return None

        with arg_scope(
                model_arg_scope(
                    weight_decay=self.model_conf['model_hyperparameter']['atten_param'].get('attention_l2_reg',
                                                                                            0.0))):
            with tf.variable_scope(name_or_scope='Share_Sequence_Layer_{}'.format(block_name),
                                   partitioner=partitioned_variables.min_max_variable_partitioner(
                                       max_partitions=self.config.get_job_config('ps_num'),
                                       min_slice_size=self.config.get_job_config('dnn_min_slice_size')),
                                   reuse=tf.AUTO_REUSE) as (scope):
                max_len = self.fg.get_seq_len_by_sequence_name(block_name)
                sequence = sequence_layer_dict[block_name]

                self.logger.info("##### sequence: {}".format(sequence))
                if self.seq_column_value is None or self.seq_column_value[block_name + '_atten_value'] is None:
                    sequence_v = None
                else:
                    if is_dcm and block_name == "buy_seq_list":
                        seq_emb_dict = self.slice_feat_emb_by_column(sequence, self.feature_columns[block_name],
                                                                     block_name + "_")
                        sequence_v = self.extract_emb_by_feat_names(seq_emb_dict,
                                                                             self.seq_column_v['dcm_user_atten_v'])
                    else:
                        seq_emb_dict = self.slice_feat_emb_by_column(sequence, self.feature_columns[block_name],
                                                                     block_name + "_")
                        sequence_v = self.extract_emb_by_feat_names(seq_emb_dict,
                                                                    self.seq_column_value[block_name + '_atten_value'])


                self.logger.info("###  sequence v is {}".format(str(sequence_v)))

                if block_name not in seq_column_len or seq_column_len[block_name] not in self.layer_dict:
                    sequence_mask = tf.sequence_mask(tf.ones_like(sequence[:, 0, 0], dtype=tf.int32), 1)
                    sequence_mask = tf.tile(sequence_mask, [1, max_len])
                else:
                    sequence_length = self.layer_dict[seq_column_len[block_name]]
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                if attention_dict is not None and block_name in attention_dict.keys():
                    attention = tf.expand_dims(attention_dict[block_name], 1)
                elif attention_layer is not None:
                    attention = tf.expand_dims(attention_layer, 1)
                else:
                    raise ValueError('Debug!! Query Attention does not have attention!!!')

                item_vec, stt_vec = atten_func(queries=attention,
                                               keys=sequence,
                                               values=sequence_v,
                                               key_masks=sequence_mask,
                                               query_masks=tf.sequence_mask(
                                                   tf.ones_like(attention[:, 0, 0], dtype=tf.int32), 1),
                                               num_units=ma_num_units,
                                               num_output_units=ma_num_output_units,
                                               scope=block_name + "_query_attention",
                                               atten_mode=self.model_conf['model_hyperparameter']['atten_param'][
                                                   'atten_mode'],
                                               reuse=tf.AUTO_REUSE,
                                               variables_collections=[self.atten_collections_dnn_hidden_layer],
                                               outputs_collections=[self.atten_collections_dnn_hidden_output],
                                               num_heads=num_heads,
                                               residual_connection=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('residual_connection', False),
                                               attention_normalize=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('attention_normalize', False),
                                               use_atten_linear_project=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('use_atten_linear_project', True))
                if self.model_conf['model_hyperparameter']['atten_param'].get('residual_connection', False):
                    ma_num_output_units = attention.get_shape().as_list()[-1]
                else:
                    ma_num_output_units = ma_num_output_units
                dec = tf.reshape(item_vec, [-1, ma_num_output_units])
                # shap config
                # if aop.interpret.shap.shap_enabled():
                #     dec = aop.interpret.shap.mark_as_input(dec,block_name,'target_attention')
                #     self.layer_dict[block_name] = dec
                # layer_dict[block_name] = dec
                # weight_dict[block_name] = stt_vec
                logger.info("Debug!!block is {},  dec is {} ".format(block_name, dec))
                dec = tf.concat(dec, axis=1)
                logger.info("Debug!!block is {}, item_vec is {}, stt_vec is {}, dec is {} ".format(block_name, item_vec,
                                                                                                   stt_vec, dec))
        logger.info("Debug!! Finish Build Query Attention: {}".format(block_name))
        return dec

    def build_self_attention(self, sequence_layer_dict, seq_column_len, name):
        logger.info("Debug!! Start Build Self Attention: {}".format(name))
        layer_dict = {}
        if sequence_layer_dict is None:
            return layer_dict
        for block_name in sequence_layer_dict.keys():
            with arg_scope(
                    model_arg_scope(weight_decay=self.model_conf['model_hyperparameter'].get('attention_l2_reg', 0.0))):
                with tf.variable_scope(name_or_scope='Share_Sequence_Layer_{}_{}'.format(name, block_name),
                                       partitioner=partitioned_variables.min_max_variable_partitioner(
                                           max_partitions=self.config.get_job_config('ps_num'),
                                           min_slice_size=self.config.get_job_config('dnn_min_slice_size')),
                                       reuse=tf.AUTO_REUSE) as (scope):
                    max_len = self.fg.get_seq_len_by_sequence_name(block_name)
                    logger.info("Debug!! max len of {} is {}".format(block_name, str(max_len)))
                    sequence = sequence_layer_dict[block_name]
                    if block_name not in seq_column_len or seq_column_len[
                        block_name] not in self.layer_dict:
                        sequence_mask = tf.sequence_mask(tf.ones_like(sequence[:, 0, 0], dtype=tf.int32), 1)
                        sequence_mask = tf.tile(sequence_mask, [1, max_len])
                    else:
                        sequence_length = self.layer_dict[seq_column_len[block_name]]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                    item_vec, stt_vec = atten_func(
                        query_masks=sequence_mask,
                        key_masks=sequence_mask,
                        queries=sequence,
                        keys=sequence,
                        num_units=self.model_conf['model_hyperparameter']['atten_param']['sa_num_units'],
                        num_output_units=self.model_conf['model_hyperparameter']['atten_param']['sa_num_output_units'],
                        activation_fn=getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
                        scope='self_attention',
                        reuse=tf.AUTO_REUSE,
                        variables_collections=[self.atten_collections_dnn_hidden_layer],
                        outputs_collections=[self.atten_collections_dnn_hidden_output],
                        num_heads=self.model_conf['model_hyperparameter']['atten_param']['num_heads'],
                        residual_connection=True,
                        attention_normalize=True
                    )
                    # if self.model_conf['model_hyperparameter']['atten_type'] == 'self':
                    # self attention
                    dec = tf.reshape(tf.reduce_mean(item_vec, axis=1), [
                        -1, self.model_conf['model_hyperparameter']['sa_num_output_units']])
                    layer_dict[block_name] = dec
                    logger.info("Debug!!item_vec is {}, stt_vec is {}".format(item_vec, stt_vec))
        logger.info("Debug!! Finsh Build Self Attention: {}".format(name))
        return layer_dict

    def build_seq_pooling_layer(self, seq_column_blocks, sequence_layer_dict, name):
        logger.info("Debug!! Start Build Pooling Layer: {}".format(name))
        if seq_column_blocks is None:
            return
        pooling_layer_dict = {}
        if len(seq_column_blocks) > 0:
            for block_name in seq_column_blocks:
                with tf.variable_scope(
                        name_or_scope=('{}_{}_Pooling_Layer').format(name, block_name),
                        partitioner=partitioned_variables.min_max_variable_partitioner(
                            max_partitions=self.config.get_job_config('ps_num'),
                            min_slice_size=self.config.get_job_config('dnn_min_slice_size')),
                        reuse=tf.AUTO_REUSE) as (scope):
                    sequence = sequence_layer_dict[block_name]
                    sequence = feedforward(sequence,
                                           num_units=[sequence.get_shape().as_list()[(-1)],
                                                      sequence.get_shape().as_list()[(-1)]],
                                           activation_fn=getActivationFunctionOp(
                                               self.model_conf.get('activation', 'lrelu')),
                                           scope='feed_forward',
                                           reuse=tf.AUTO_REUSE)
                    pooling_layer_dict[block_name] = tf.reduce_mean(sequence, axis=1)
        logger.info("Debug!! Finsh Build Pooling Layer: {}".format(name))
        return pooling_layer_dict

    def build_sequence(self, seq_column_blocks, seq_column_len, name):
        features = self.features
        feature_columns = self.feature_columns
        sequence_layer_dict = {}
        if seq_column_blocks is None or len(seq_column_blocks) == 0:
            return
        with tf.variable_scope(name_or_scope='{}_seq_input_from_feature_columns'.format(name),
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config('ps_num'),
                                   min_slice_size=self.config.get_job_config('embedding_min_slice_size')),
                               reuse=tf.AUTO_REUSE) as (scope):
            if len(seq_column_blocks) > 0:
                for block_name in seq_column_blocks:
                    logger.info("Debug seq_column_bolcks: {}".format(name, block_name))
                    if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                        raise ValueError('block_name:(%s) not in feature_columns for seq' % block_name)
                    seq_len = self.fg.get_seq_len_by_sequence_name(block_name)

                    sequence_stack = _input_from_feature_columns(features,
                                                                 feature_columns[block_name],
                                                                 weight_collections=None,
                                                                 trainable=True,
                                                                 scope=scope,
                                                                 output_rank=3,default_name='sequence_input_from_feature_columns')
                    sequence_stack = tf.reshape(sequence_stack, [-1, seq_len, sequence_stack.get_shape()[(-1)].value])
                    sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])

                    if block_name in seq_column_len and seq_column_len[block_name] in self.layer_dict:
                        sequence_length = self.layer_dict[seq_column_len[block_name]]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), seq_len)
                        sequence_stack = tf.reshape(
                            tf.where(tf.reshape(sequence_mask, [-1]), sequence_2d, tf.zeros_like(sequence_2d)),
                            tf.shape(sequence_stack))
                    else:
                        sequence_stack = tf.reshape(sequence_2d, tf.shape(sequence_stack))
                    sequence_layer_dict[block_name] = sequence_stack
        return sequence_layer_dict

    def extract_emb_by_feat_names(self, emb_dict, feat_names):
        self.logger.info("extract emb_{}, by feat_names: {}".format(emb_dict, feat_names))
        emb_list = []
        for feat_name in feat_names:
            if feat_name in emb_dict:
                emb_list.append(emb_dict[feat_name])
        emb = tf.concat(emb_list, -1)
        return emb

    def slice_feat_emb_by_column(self, emb, feature_columns, remove_prefix=None):
        i = 0
        feat_emb_dict = {}
        length = len(emb.shape)
        for column in sorted(set(feature_columns), key=lambda x: x.key):
            if isinstance(column, _RealValuedColumn):
                feat_name = column.column_name
            elif isinstance(column, _EmbeddingColumn):
                feat_name = column.sparse_id_column.column_name
            else:
                feat_name = column.source_column.column_name
            if remove_prefix:
                feat_name = feat_name.replace(remove_prefix, "")
            self.logger.info("##### slice_feat_emb column feat_name: {}".format(feat_name))
            try:
                dim = column.dimension
            except:
                dim = column.embedding_dimension
            if length == 2:
                feat_emb_dict[feat_name] = emb[:, i:i + dim]
            elif length == 3:
                feat_emb_dict[feat_name] = emb[:, :, i:i + dim]
            i += dim
        return feat_emb_dict
