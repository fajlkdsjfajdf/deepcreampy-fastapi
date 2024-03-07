import onnx
import tf2onnx
from tf2onnx.graph import GraphUtil, Graph, Node
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from onnx.numpy_helper import from_array
import tensorflow as tf
import numpy as np


def extract_image_patches(sizes, strides, rates, padding):
    # Alternative implementation constraints.
    assert len(sizes) == 4 and sizes[1] == sizes[2]
    assert strides == [1, 1, 1, 1]
    assert rates == [1, 1, 1, 1]
    patch_size = sizes[1]

    @tf.function
    def function(tensor):
        # Prepare tensor.
        tensor = tf.transpose(tensor, perm=[0, 3, 1, 2])
        tensor = tf.expand_dims(tensor, -1)

        # Create an identity kernel.
        kernel = tf.reshape(tf.eye(patch_size ** 2), [patch_size, patch_size, 1, patch_size ** 2])

        # Convolve with identity kernel.
        patches_simulation = tf.nn.conv2d(tensor, kernel, strides=[1, 1, 1, 1], padding=padding)
        patches_simulation = tf.transpose(patches_simulation, perm=[0, 2, 3, 4, 1])
        patches_simulation_shape = tf.shape(patches_simulation)

        return tf.reshape(patches_simulation, [
            patches_simulation_shape[0],
            patches_simulation_shape[1],
            patches_simulation_shape[2],
            -1,
        ])

    return function


def insert_graph(g: Graph, copy: Graph, input_name):
    copy.topological_sort(copy.get_nodes())
    assert len(copy.input_names) == 1
    new_output_names = {copy.input_names[0]: input_name}

    for node in copy.get_nodes():
        if node.type == "Placeholder":
            continue

        inputs = [new_output_names[name] for name in node.input]
        new_node = g.make_node(node.type, inputs, attr=node.attr,
                               shapes=node.output_shapes, dtypes=node.output_dtypes)

        assert len(node.output) == 1
        new_output_names[node.output[0]] = new_node.output[0]

    assert len(copy.outputs) == 1
    return new_output_names[copy.outputs[0]]


def rewrite_extract_image_patches(g: Graph, ops):
    pattern = OpTypePattern("ExtractImagePatches", name="extract")
    matches = GraphMatcher(pattern).match_ops(ops)
    if not matches:
        return

    for match in matches:
        node: Node = match.get_op("extract")
        f = extract_image_patches(
            sizes=node.get_attr_value("ksizes"),
            strides=node.get_attr_value("strides"),
            rates=node.get_attr_value("rates"),
            padding=node.get_attr_str("padding"),
        )

        # Input signature after model pruning.
        input_signature = [np.empty([1, 32, 32, 256], dtype=np.float32)]
        f_model, _ = tf2onnx.convert.from_function(f, input_signature=input_signature)
        f_graph = GraphUtil.create_graph_from_onnx_model(f_model)

        output_name = insert_graph(g, f_graph, node.input[0])
        g.replace_all_inputs(node.output[0], output_name)
        g.remove_node(node.name)


def prune_onnx_model(model):
    def find_node_by_name(nodes, name):
        for node in nodes:
            if node.name == name:
                return node

    def find_initializer_by_name(graph, name):
        for initializer in graph.initializer:
            if initializer.name == name:
                return initializer

    # Prune duplicated network for batching.
    nodes = model.graph.node
    block_end = find_node_by_name(nodes, "CB1/concat_6")
    del block_end.input[1:]

    # Reshape for batch size of 1.
    reshape = find_node_by_name(nodes, "CB1/Reshape_1")
    reshape_const = find_initializer_by_name(model.graph, reshape.input[1])
    reshape_const.CopyFrom(from_array(np.int64([1, 1, 1, 900, 2304]), reshape_const.name))

    # Change external dimensions.
    for node in list(model.graph.input) + list(model.graph.output):
        node.type.tensor_type.shape.dim[0].dim_value = 1

    # Dangling nodes must be eliminated.
    model = GraphUtil.optimize_model_proto(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def convert_model(checkpoint, target):
    inputs = ["Placeholder:0", "Placeholder_1:0", "Placeholder_2:0"]
    outputs = ["add:0"]

    graph, inputs, outputs = tf2onnx.tf_loader.from_checkpoint(checkpoint, inputs, outputs)
    onnx_graph, _ = tf2onnx.convert.from_graph_def(
        graph, input_names=inputs, output_names=outputs,
        custom_rewriter=[rewrite_extract_image_patches],
    )

    with open(target, "wb") as f:
        model = prune_onnx_model(onnx_graph)
        f.write(model.SerializeToString())


def main():
    convert_model("./models/bar/Train_775000.meta", "./vendor/bar.onnx")
    convert_model("./models/mosaic/Train_290000.meta", "./vendor/mosaic.onnx")


if __name__ == '__main__':
    main()
