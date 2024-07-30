# %% [markdown]
# # Proyek Pertama: Pengembangan Machine Learning Pipeline

# %% [markdown]
# - Nama: Nicolas Debrito
# - Email: nicolas.debrito66@gmail.com
# - Id Dicoding: reezzy

# %% [markdown]
# ## Import Library

# %% [markdown]
# Menyiapkan libaray yang akan digunakan

# %%
import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma
import pandas as pd
import os

# %% [markdown]
# ## Load Data

# %% [markdown]
# Membuat dataframe untuk data spam

# %%
df = pd.read_csv('data/Spam_Detection.csv')
df.head()

# %% [markdown]
# Mengubah Label menjadi numerikal dengan ham 0 (ham) dan 1 (spam)

# %%
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'ham' else 1)
df.head()

# %% [markdown]
# Menyimpan data yang telah diubah dalam csv

# %%
df.to_csv('fix_data/Fix_Data.csv', index=False)

# %% [markdown]
# ## Set Variable

# %% [markdown]
# Menyiapkan variable konstanta untuk nama nama path yang digunakan

# %%
PIPELINE_NAME = "spam-pipeline"
SCHEMA_PIPELINE_NAME = "spam-tfdv-schema"

PIPELINE_ROOT = os.path.join('reezzy-pipelines', PIPELINE_NAME)

METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

DATA_ROOT = "fix_data"

# %% [markdown]
# Menyiapkan InteractiveContext

# %%
interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)

# %% [markdown]
# Membuat ExampleGen untuk csv

# %%
output = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
    ])
)

example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)

# %% [markdown]
# Menjalankan interactive_context untuk example_gen

# %%
interactive_context.run(example_gen)

# %% [markdown]
# Membuat StatisticsGen dari example_gen dan menjalankan interactive_context

# %%
statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

interactive_context.run(statistics_gen)

# %% [markdown]
# Melihat summary dari staistics_gen

# %%
interactive_context.show(statistics_gen.outputs["statistics"])

# %% [markdown]
# Membuat SchemaGen dari statistics_gen dan menjalankan interactive_context

# %%
schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

interactive_context.run(schema_gen)

# %% [markdown]
# Melihat schema yang dibuat

# %%
interactive_context.show(schema_gen.outputs["schema"])

# %% [markdown]
# Membuat ExampleValidator dengan statistics_gen dan schema_gen dan menjalankan interactive_context

# %%
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
interactive_context.run(example_validator)

# %% [markdown]
# Melihat anomali dari validator

# %%
interactive_context.show(example_validator.outputs['anomalies'])

# %% [markdown]
# Mendeklarasikan file transform

# %%
TRANSFORM_MODULE_FILE = "spam_transform.py"

# %% [markdown]
# Membuat file transform

# %%
%%writefile {TRANSFORM_MODULE_FILE}

import tensorflow as tf

LABEL_KEY = "Label"
FEATURE_KEY = "Mail"

def transformed_name(key):
    return key + "_xf"

def preprocessing_fn(inputs):
    outputs = {}
    
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs

# %% [markdown]
# Melakuakn Transform dari example_gen, schema_gen dan modul transform serta menjalankan interactive_context

# %%
transform  = Transform(
    examples=example_gen.outputs['examples'],
    schema= schema_gen.outputs['schema'],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)

interactive_context.run(transform)

# %% [markdown]
# Mendeklarasiakn file trainer

# %%
TRAINER_MODULE_FILE = "spam_trainer.py"

# %% [markdown]
# Membuat file trainer

# %%
%%writefile {TRAINER_MODULE_FILE}

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "Label"
FEATURE_KEY = "Mail"

def transformed_name(key):
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern,
            tf_transform_output,
            num_epochs,
            batch_size=64)->tf.data.Dataset:

    transform_feature_spec = (tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
		file_pattern=file_pattern,
		batch_size=batch_size,
		features=transform_feature_spec,
		reader=gzip_reader_fn,
		num_epochs=num_epochs,
		label_key=transformed_name(LABEL_KEY)
    )

    return dataset

VOCAB_SIZE = 1000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
	standardize="lower_and_strip_punctuation",
	max_tokens=VOCAB_SIZE,
	output_mode='int',
	output_sequence_length=SEQUENCE_LENGTH
)

embedding_dim=16
def model_builder():
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
		loss = 'binary_crossentropy',
		optimizer=tf.keras.optimizers.Adam(0.01),
		metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):

        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args:FnArgs) -> None:

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
		log_dir = log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10).repeat(10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10).repeat(10)
    vectorize_layer.adapt(
		[j[0].numpy()[0] for j in [
			i[0][transformed_name(FEATURE_KEY)]
			for i in list(train_set)
		]]
    )

    model = model_builder()

    model.fit(x = train_set,
    validation_data = val_set,
    callbacks = [tensorboard_callback, es, mc],
    steps_per_epoch=100,
    validation_steps=100,
    epochs=10)

    signatures = {
		'serving_default':
		_get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
			tf.TensorSpec(
				shape=[None],
				dtype=tf.string,
				name='examples'
			)
		)
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

# %% [markdown]
# Menjalankan Trainer dengan modul trainer, hasil transform, transform graph, shcema, data train, dan data eval

# %%
trainer = Trainer(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)

interactive_context.run(trainer)

# %% [markdown]
# Membuat Resolver dan menjalankan interactive_context 

# %%
model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')

interactive_context.run(model_resolver)

# %% [markdown]
# Membuat konfigurasi untuk evaluator

# %%
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='Label')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                            threshold=tfma.MetricThreshold(
                                value_threshold=tfma.GenericValueThreshold(
                                    lower_bound={'value':0.5}
                                ),
                                change_threshold=tfma.GenericChangeThreshold(
                                    direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                    absolute={'value':0.0001}
                                )
                            ))
        ])
    ]
)

# %% [markdown]
# Membuat Evaluator dengan dataset, model, baseline_model, dan konfigurasi evaluator

# %%
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)

interactive_context.run(evaluator)

# %% [markdown]
# Memvisualisasikan menggunakan library TFMA

# %%
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(tfma_result)

# %% [markdown]
# Membuat komponen Pusher

# %%
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model_dir/spam-detection-model'
		)
	)
)

interactive_context.run(pusher)


