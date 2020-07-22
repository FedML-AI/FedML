package com.example.ondevicetraining;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

//import androidx.appcompat.app.AppCompatActivity;
//import androidx.core.app.ActivityCompat;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.Pooling2D;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sin;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;

import lombok.NonNull;

public class MainActivity extends AppCompatActivity {
    //读写权限
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};
    private static int REQUEST_PERMISSION_CODE = 1; //请求状态码

    // reddit model layers
    private String EMBEDDING = "embedding", LSTM = "lstm", RNN_OUTPUT = "rnn_output";
    // entire model
    private String
            CELEBA = "celeba",
            REDDIT = "reddit",
            FEMNIST = "femnist",
            SHAKESPEARE = "shakespeare",
            SENT140 = "sent140", SENT140_RNN = "sent140_stacked_lstm", SENT140_REG = "sent140_bag_log_reg";
    //celeba model layers
    private String CONVOLUTION_2D = "convolution_2d", OUTPUT = "output";
    private String BATCH_NORMALIZATION = "batch_normalization", POOLING_2D = "pooling_2d";

    private File IRIS_DIR = new File(Environment.getExternalStorageDirectory(), "iris_classifier");
    private File LOOKUP_TABLE_REDDIT_DIR = new File(IRIS_DIR, "lookup_table_reddit");
    private File LOOKUP_TABLE_CELEBA_DIR = new File(IRIS_DIR, "lookup_table_celeba");

    //layer of reddit model
    private File EMBEDDING_DIR = new File(LOOKUP_TABLE_REDDIT_DIR, EMBEDDING);
    private File LSTM_DIR = new File(LOOKUP_TABLE_REDDIT_DIR, LSTM);
    private File RNN_OUTPUT_DIR = new File(LOOKUP_TABLE_REDDIT_DIR, RNN_OUTPUT);

    //layer of celeba model
    private File CONVOLUTION_2D_DIR = new File(LOOKUP_TABLE_CELEBA_DIR, CONVOLUTION_2D);
    private File BATCH_NORMALIZATION_DIR = new File(LOOKUP_TABLE_CELEBA_DIR, BATCH_NORMALIZATION);
    private File POOLING_2D_DIR = new File(LOOKUP_TABLE_CELEBA_DIR, POOLING_2D);
    private File OUTPUT_DIR = new File(LOOKUP_TABLE_CELEBA_DIR, OUTPUT);

    private File REDDIT_DIR = new File(IRIS_DIR, REDDIT);
    private File CELEBA_DIR = new File(IRIS_DIR, CELEBA);
    private File FEMNIST_DIR = new File(IRIS_DIR, FEMNIST);
    private File SHAKESPEARE_DIR = new File(IRIS_DIR, SHAKESPEARE);
    private File SENT140_DIR = new File(IRIS_DIR, SENT140),
            SENT140_RNN_DIR = new File(SENT140_DIR, SENT140_RNN),
            SENT140_REG_DIR = new File(SENT140_DIR, SENT140_REG);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
            }
        }
        make_dir();
        // run initialize(String) to avoid cold boot
        initialize("rnn");
        System.out.println("finish initialize");

//        MyServer myServer = new MyServer(8080);
//        try {
//            myServer.start();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        onCLick();

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION_CODE) {
            for (int i = 0; i < permissions.length; i++) {
                Log.i("MainActivity", "permission：" + permissions[i] + ", result: " + grantResults[i]);
            }
        }
    }

    private void make_dir() {
        // top dir
        if (!IRIS_DIR.exists()) {
            IRIS_DIR.mkdirs();
        }
        if (!LOOKUP_TABLE_REDDIT_DIR.exists()) {
            LOOKUP_TABLE_REDDIT_DIR.mkdirs();
        }
        if (!LOOKUP_TABLE_CELEBA_DIR.exists()) {
            LOOKUP_TABLE_CELEBA_DIR.mkdirs();
        }

        // lookup-table-reddit layer dir
        if (!EMBEDDING_DIR.exists()) {
            EMBEDDING_DIR.mkdirs();
        }
        if (!LSTM_DIR.exists()) {
            LSTM_DIR.mkdirs();
        }
        if (!RNN_OUTPUT_DIR.exists()) {
            RNN_OUTPUT_DIR.mkdirs();
        }

        // entire model dir
        if (!REDDIT_DIR.exists()) {
            REDDIT_DIR.mkdirs();
        }
        if (!CELEBA_DIR.exists()) {
            CELEBA_DIR.mkdirs();
        }
        if (!FEMNIST_DIR.exists()) {
            FEMNIST_DIR.mkdirs();
        }
        if (!SHAKESPEARE_DIR.exists()) {
            SHAKESPEARE_DIR.mkdirs();
        }
        if (!SENT140_DIR.exists()) {
            SENT140_DIR.mkdirs();
        }
        if (!SENT140_RNN_DIR.exists()) {
            SENT140_RNN_DIR.mkdirs();
        }
        if (!SENT140_REG_DIR.exists()) {
            SENT140_REG_DIR.mkdirs();
        }

        // lookup-table-celeba layer dir
        if (!OUTPUT_DIR.exists()) {
            OUTPUT_DIR.mkdirs();
        }
        if (!CONVOLUTION_2D_DIR.exists()) {
            CONVOLUTION_2D_DIR.mkdirs();
        }
        if (!BATCH_NORMALIZATION_DIR.exists()) {
            BATCH_NORMALIZATION_DIR.mkdirs();
        }
        if (!POOLING_2D_DIR.exists()) {
            POOLING_2D_DIR.mkdirs();
        }
    }

    private void onCLick() {
        Button button = findViewById(R.id.button_500);
        button.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                try {
                    // measure the on-device trianing time of pre-defined DNN
                    // modify the line below if needed, e.g. olaf_celeba();
                    // modify the hyper-parameter if needed
                    // note that we only measure the training time of one batch, so the optimizer, params, initializer do not matter.
                    // functions we used are olaf_*
                    olaf_reddit();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

    }

    private void olaf_femnist() throws Exception {
        int num_class = 62;
        Random random = new Random();
        int num_iteration = 10;
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(0.1))
                .list();
        listBuilder = listBuilder
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.RELU).kernelSize(5, 5).stride(1, 1).nOut(32).build())
                .layer(new Pooling2D.Builder().poolingType(PoolingType.MAX).stride(2, 2).kernelSize(2, 2).build())
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.RELU).kernelSize(5, 5).stride(1, 1).nOut(64).build())
                .layer(new Pooling2D.Builder().poolingType(PoolingType.MAX).stride(2, 2).kernelSize(2, 2).build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(2048).build())
                .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).nOut(num_class).build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1));
        MultiLayerNetwork net = new MultiLayerNetwork(listBuilder.backpropType(BackpropType.Standard).build());
        net.init();
        for (int batch_size = 1; batch_size <= 20; batch_size++) {
            INDArray features = Nd4j.randn(batch_size, 1, 28, 28); //c h w
            INDArray labels = Nd4j.create(batch_size, 1);
            for (int i = 0; i < batch_size; i++) {
                labels.putScalar(new int[]{i, 0}, random.nextInt(num_class));
            }
            long[] total_cost = new long[num_iteration];
            for (int i = 0; i < num_iteration; i++) {
                long start = new Date().getTime();
                net.fit(features, labels);
                long end = new Date().getTime();
                total_cost[i] = end - start;
            }
            String filename = FEMNIST + "_" + batch_size + ".txt";
            File save_location = new File(FEMNIST_DIR, filename);
            BufferedWriter writer = new BufferedWriter(new FileWriter(save_location));
            writer.write(Arrays.toString(total_cost).replaceAll(" ", ""));
            writer.close();
            System.out.println(save_location);
        }
    }

    private void olaf_shakespeare() throws Exception {
        double learning_rate = 0.0003;
        int hidden_size = 256;
        int seq_len = 80;
        int vocab_size = 80;
        int num_iteration = 10;
        Random random = new Random();

        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(learning_rate))
                .list()
                .layer(new EmbeddingSequenceLayer.Builder().nIn(vocab_size).nOut(8).build())
                .layer(new LSTM.Builder().nIn(8).nOut(hidden_size).build())
                .layer(new LastTimeStep(new LSTM.Builder().nIn(hidden_size).nOut(hidden_size).build()))
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
                        .nIn(hidden_size).nOut(vocab_size).activation(Activation.SOFTMAX).build());
        MultiLayerConfiguration conf = listBuilder.backpropType(BackpropType.Standard).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        for (int batch_size = 5; batch_size <= 100; batch_size += 5) {
            INDArray features = Nd4j.create(batch_size, seq_len);
            INDArray labels = Nd4j.create(batch_size, 1);
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < seq_len; j++) {
                    features.putScalar(new int[]{i, j}, random.nextInt(vocab_size));
                }
                labels.putScalar(new int[]{i, 0}, random.nextInt(vocab_size));
            }
            long[] total_cost = new long[num_iteration];
            for (int i = 0; i < num_iteration; i++) {
                long start = new Date().getTime();
                net.fit(features, labels);
                total_cost[i] = new Date().getTime() - start;
            }
            String filename = SHAKESPEARE + "_" + batch_size + ".txt";
            File locationToSave = new File(SHAKESPEARE_DIR, filename);
            BufferedWriter out;
            out = new BufferedWriter(new FileWriter(locationToSave));
            out.write(Arrays.toString(total_cost).replaceAll(" ", ""));
            out.close();
            System.out.println(locationToSave);
        }
    }


    private void olaf_celeba() throws Exception {
        int num_class = 2;
        Random random = new Random();
        int num_iteration = 3;
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(0.1))
                .list();
        for (int i = 0; i < 3; i++) {
            listBuilder = listBuilder
                    .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(32).build())
                    .layer(new BatchNormalization.Builder().build())
                    .layer(new Pooling2D.Builder().poolingType(PoolingType.MAX).stride(2, 2).build())
                    .layer(new ActivationLayer.Builder().activation(Activation.RELU).build());
        }
        listBuilder = listBuilder
                .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX).
                        lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).nOut(num_class).build())
                .setInputType(InputType.convolutionalFlat(84, 84, 3));
        MultiLayerConfiguration conf = listBuilder
                .backpropType(BackpropType.Standard)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        for (int batch_size = 1; batch_size <= 10; batch_size++) {
            INDArray features = Nd4j.randn(batch_size, 3, 84, 84); //c h w
            INDArray labels = Nd4j.create(batch_size, 1);
            for (int i = 0; i < batch_size; i++) {
                labels.putScalar(new int[]{i, 0}, random.nextInt(num_class));
            }
            long[] total_cost = new long[num_iteration];
            for (int i = 0; i < num_iteration; i++) {
                long start = new Date().getTime();
//                net.fit(features, labels);
                System.out.println(Arrays.toString(net.output(features).shape()));
                long end = new Date().getTime();
                total_cost[i] = end - start;
            }
            String filename = CELEBA + "_" + batch_size + "_" + num_class + ".txt";
            File save_location = new File(CELEBA_DIR, filename);
            BufferedWriter writer = new BufferedWriter(new FileWriter(save_location));
            writer.write(Arrays.toString(total_cost).replaceAll(" ", ""));
            writer.close();
            System.out.println(save_location);
        }
    }

    private void olaf_reddit() throws Exception {
        //fixed params
        int batch_size = 10;
        int vocab_size = 10000;
        double learning_rate = 0.0003;
        int nLayers = 2;
        int size = 256;
        int seq_len = 10;
//        int batch_size = 1;
        Random random = new Random();
        int num_iteration = 10;


        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(learning_rate))
                .list()
                .layer(new EmbeddingSequenceLayer.Builder()
                        .nIn(vocab_size).nOut(size).build());
        for (int i = 0; i < nLayers; i++) {
            listBuilder = listBuilder
                    .layer(new LSTM.Builder().nIn(size).nOut(size).build());
//                    .layer(new DropoutLayer.Builder(keepProb).build());
        }
        listBuilder = listBuilder.layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
                .nIn(size).nOut(vocab_size)
                .activation(Activation.SOFTMAX)
                .build());
        MultiLayerConfiguration conf = listBuilder.backpropType(BackpropType.Standard).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray features = Nd4j.create(batch_size, seq_len);
        INDArray labels = Nd4j.create(batch_size, 1, seq_len);
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < seq_len; j++) {
                features.putScalar(new int[]{i, j}, random.nextInt(vocab_size));
                labels.putScalar(new int[]{i, 0, j}, random.nextInt(vocab_size));
            }

        }
        long[] total_cost = new long[num_iteration];
        for (int i = 0; i < num_iteration; i++) {
            long start = new Date().getTime();
//            net.output(features);
            net.fit(features, labels);
            total_cost[i] = new Date().getTime() - start;
        }
        String filename = REDDIT + "_" + batch_size + ".txt";
        File locationToSave = new File(REDDIT_DIR, filename);
        BufferedWriter out = new BufferedWriter(new FileWriter(locationToSave));
        out.write(Arrays.toString(total_cost));
        out.close();
        System.out.println(locationToSave + "\t" + Arrays.toString(total_cost));

    }

    private void olaf_sent140_rnn() throws Exception {
        double learning_rate = 0.0003;
        int num_class = 2;
        int size = 100;
        int seq_len = 25;
        Random random = new Random();
        int num_iteration = 10;

        int max_batch_size = 20, delta_batch_size = 2;
        int max_vocab_size = 10000, delta_vocab_size = 1000;
        int batch_size, vocab_size;

        File recover_sent140_rnn = new File(SENT140_RNN_DIR, "recover_sent140_rnn.txt");
        if (!recover_sent140_rnn.exists()) {
            BufferedWriter out = new BufferedWriter(new FileWriter(recover_sent140_rnn));
            out.write(delta_vocab_size + "\n");
            out.write(delta_batch_size + "\n");
            out.close();
        }
        BufferedReader reader = new BufferedReader(new FileReader(recover_sent140_rnn));
        vocab_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
        batch_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));

        while (vocab_size <= max_vocab_size && batch_size <= max_batch_size) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new AdaGrad(learning_rate))
                    .list()
                    .layer(new EmbeddingSequenceLayer.Builder().nIn(vocab_size + 1).nOut(size).build())
                    .layer(new LSTM.Builder().nIn(size).nOut(size).build())
                    .layer(new LastTimeStep(new LSTM.Builder().nIn(size).nOut(size).build()))
                    .layer(new DenseLayer.Builder().nIn(size).nOut(128).build())
                    .layer(new OutputLayer.Builder().nIn(128).nOut(num_class)
                            .activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).build())
                    .backpropType(BackpropType.Standard)
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray features = Nd4j.create(batch_size, seq_len);
            INDArray labels = Nd4j.create(batch_size, 1);
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < seq_len; j++) {
                    features.putScalar(new int[]{i, j}, random.nextInt(vocab_size));
                }
                labels.putScalar(new int[]{i, 0}, random.nextInt(vocab_size));
            }
            long[] total_cost = new long[num_iteration];
            for (int i = 0; i < num_iteration; i++) {
                long start = new Date().getTime();
//            net.output(features);
                net.fit(features, labels);
                total_cost[i] = new Date().getTime() - start;
            }
            String filename = SENT140_RNN +
                    "_" + vocab_size +
                    "_" + batch_size + ".txt";
            File locationToSave = new File(SENT140_RNN_DIR, filename);
            BufferedWriter out = new BufferedWriter(new FileWriter(locationToSave));
            out.write(Arrays.toString(total_cost));
            out.close();
            System.out.println(locationToSave + "\t" + Arrays.toString(total_cost));

            if (batch_size < max_batch_size) {
                batch_size += delta_batch_size;
            } else {
                batch_size = delta_batch_size;
                vocab_size += delta_vocab_size;
            }
//            if (vocab_size < max_vocab_size) {
//                vocab_size += delta_vocab_size;
//            } else {
//                vocab_size = delta_vocab_size;
//                batch_size += delta_batch_size;
//            }
            out = new BufferedWriter(new FileWriter(recover_sent140_rnn));
            out.write(vocab_size + "\n");
            out.write(batch_size + "\n");
            out.close();
        }
    }

    private void olaf_sent140_reg() throws Exception {
        double learning_rate = 0.0003;
        int num_class = 2;
        Random random = new Random();
        int num_iteration = 10;

        int max_batch_size = 20, delta_batch_size = 2;
        int max_vocab_size = 10000, delta_vocab_size = 1000;
        int batch_size, vocab_size;

        File recover_sent140_reg = new File(SENT140_REG_DIR, "recover_sent140_reg.txt");
        if (!recover_sent140_reg.exists()) {
            BufferedWriter out = new BufferedWriter(new FileWriter(recover_sent140_reg));
            out.write(delta_vocab_size + "\n");
            out.write(delta_batch_size + "\n");
            out.close();
        }
        BufferedReader reader = new BufferedReader(new FileReader(recover_sent140_reg));
        vocab_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
        batch_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));

        while (vocab_size <= max_vocab_size && batch_size <= max_batch_size) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new AdaGrad(learning_rate))
                    .list()
                    .layer(new OutputLayer.Builder().nIn(vocab_size).nOut(num_class)
                            .activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).build())
                    .backpropType(BackpropType.Standard)
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray features = Nd4j.randn(batch_size, vocab_size);
            INDArray labels = Nd4j.create(batch_size, 1);
            for (int i = 0; i < batch_size; i++) {
                labels.putScalar(new int[]{i, 0}, random.nextInt(num_class));
            }
            long[] total_cost = new long[num_iteration];
            for (int i = 0; i < num_iteration; i++) {
                long start = new Date().getTime();
//            net.output(features);
                net.fit(features, labels);
                total_cost[i] = new Date().getTime() - start;
            }
            String filename = SENT140_REG +
                    "_" + vocab_size +
                    "_" + batch_size + ".txt";
            File locationToSave = new File(SENT140_REG_DIR, filename);
            BufferedWriter out = new BufferedWriter(new FileWriter(locationToSave));
            out.write(Arrays.toString(total_cost));
            out.close();
            System.out.println(locationToSave + "\t" + Arrays.toString(total_cost));

            if (batch_size < max_batch_size) {
                batch_size += delta_batch_size;
            } else {
                batch_size = delta_batch_size;
                vocab_size += delta_vocab_size;
            }
            out = new BufferedWriter(new FileWriter(recover_sent140_reg));
            out.write(vocab_size + "\n");
            out.write(batch_size + "\n");
            out.close();
            if (batch_size == delta_batch_size) {
                break;
            }
        }
    }

    private void mobileNet(int batch_size, int[] input_shape) {
        // input_shape = {h, w, c}
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.1, 0.9))
                .list();

        listBuilder = listBuilder
                //1
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(2, 2).nOut(32).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //2
                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(32).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //3
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(64).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //4
                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(2, 2).nOut(64).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //5
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(128).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //6
                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(128).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //7
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(128).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //8
                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(2, 2).nOut(128).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //9
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(256).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //10
                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(256).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //11
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(256).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //12
                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(2, 2).nOut(256).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                //13
                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(512).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build());

        for (int i = 0; i < 5; i++) {
            listBuilder = listBuilder
                    .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(512).depthMultiplier(1).build())
                    .layer(new BatchNormalization.Builder().build())
                    .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())

                    .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(512).build())
                    .layer(new BatchNormalization.Builder().build())
                    .layer(new ActivationLayer.Builder().activation(Activation.RELU).build());
        }

        listBuilder = listBuilder
                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(2, 2).nOut(512).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())

                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(1024).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())

                .layer(new SeparableConvolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(1024).depthMultiplier(1).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())

                .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(1, 1).stride(1, 1).nOut(1024).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())

                .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                .layer(new DenseLayer.Builder().activation(Activation.SOFTMAX).nOut(100).build())
                .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).nOut(10).build())
                .setInputType(InputType.convolutionalFlat(input_shape[0], input_shape[1], input_shape[2]));


        MultiLayerConfiguration conf = listBuilder
                .backpropType(BackpropType.Standard)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray features = Nd4j.create(batch_size, input_shape[2], input_shape[1], input_shape[0]); //c h w
        INDArray labels = Nd4j.create(batch_size, 1);

        DataSet allData = new DataSet(features, labels);
        List<DataSet> list = allData.asList();
        Collections.shuffle(list, new Random());
        DataSetIterator iterator = new ListDataSetIterator(list, batch_size);

        long start = new Date().getTime();
        net.fit(iterator, 1);
        long end = new Date().getTime();
        long cost = end - start;

        String filename = "MLN_mobileNet.txt";
        File iris_dir = new File(Environment.getExternalStorageDirectory(), "iris_classifier");
        if (!iris_dir.exists()) {
            iris_dir.mkdirs();
        }
        File locationToSave = new File(iris_dir, filename);
        try {
            FileWriter writer = new FileWriter(locationToSave, true);
            writer.write("batch_size=" + batch_size + "\tinput_shape=" + Arrays.toString(input_shape) + "\tcost=" + cost + "ms\n");
            writer.close();
            System.out.println("success: " + locationToSave);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void ptbModel(int maxSize) {
        Random random = new Random();
        double learningRate = random.nextDouble();
        double momentum = random.nextDouble();
        double keepProb = random.nextDouble();
        int vocabSize = maxSize;
        int seq_len = 1;
        int seed = 1024;
        int embenddingSize = random.nextInt(maxSize) + 1;
        int batchSize = random.nextInt(20) + 1;
        int nLayers = random.nextInt(4) + 1;
        int[] nHidden = new int[nLayers + 1];
        nHidden[0] = embenddingSize;
        for (int i = 1; i <= nLayers; i++) {
            nHidden[i] = random.nextInt(1024) + 1;
        }


        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, momentum))
                .list()
                .layer(new EmbeddingLayer.Builder()
                        .nIn(vocabSize).nOut(embenddingSize).build());
        for (int i = 1; i <= nLayers; i++) {
            listBuilder = listBuilder
                    .layer(new LSTM.Builder().nIn(nHidden[i - 1]).nOut(nHidden[i]).build())
                    .layer(new DropoutLayer.Builder(keepProb).build());
        }
        listBuilder = listBuilder.layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
                .nIn(nHidden[nLayers]).nOut(vocabSize)
                .activation(Activation.SOFTMAX)
                .build());
        MultiLayerConfiguration conf = listBuilder.backpropType(BackpropType.Standard).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        INDArray features = Nd4j.create(batchSize * seq_len, 1);
        INDArray labels = Nd4j.create(batchSize * seq_len, 1, 1);
        for (int i = 0; i < batchSize * seq_len; i++) {
            features.putScalar(new int[]{i, 0}, random.nextInt(vocabSize));
            labels.putScalar(new int[]{i, 0, 0}, random.nextInt(vocabSize));
        }
        DataSet allData = new DataSet(features, labels);
        List<DataSet> list = allData.asList();
        Collections.shuffle(list, new Random());
        DataSetIterator iterator = new ListDataSetIterator(list, batchSize);

        while (true) {
            long start = new Date().getTime();
            net.fit(iterator, 1);
            long end = new Date().getTime();
            long cost = end - start;

            String filename = "MLN_ptbModel_" + end + "_" + cost + ".txt";
            File iris_dir = new File(Environment.getExternalStorageDirectory(), "iris_classifier");
//            File iris_dir = new File(getFilesDir(), "iris_classifier");
            if (!iris_dir.exists()) {

                iris_dir.mkdirs();
            }
            File locationToSave = new File(iris_dir, filename);
            BufferedWriter out;
            try {
                out = new BufferedWriter(new FileWriter(locationToSave));
                out.write("seed " + seed + "\n");
                out.write("learningRate " + learningRate + "\n");
                out.write("momentum " + momentum + "\n");
                out.write("keepProb " + keepProb + "\n");
                out.write("vocabSize " + vocabSize + "\n");
                out.write("batchSize " + batchSize + "\n");
                out.write("embenddingSize " + embenddingSize + "\n");
                out.write("nLayers " + nLayers + "\n");
                out.write("nHidden " + Arrays.toString(nHidden) + "\n");
                out.write("# nHidden[0] == [embenddingSize], nHidden[1] to nHidden[nLayers] is the real hidden_size used in nLayers LSTM\n");
                out.write("# vocabSize, embenddingSize, seed range from 1 to input_max_size ([1e3, 1e4])\n");
                out.write("# nLayers in [1, 4], nHidden in [1, 1024], batchSize in [1, 100]\n");
                out.close();
                System.out.println("success " + maxSize + ": " + locationToSave);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }


    private void initialize(String layer) {
        int batch_size = 10;
        int num_iteration = 10;
        Random random = new Random();
        if (layer.equals("cnn")) {
            int num_class = 10;
            NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(0.1, 0.9))
                    .list();
            listBuilder = listBuilder
                    .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(32).build())
                    .layer(new BatchNormalization.Builder().build())
                    .layer(new Pooling2D.Builder().poolingType(PoolingType.MAX).stride(2, 2).build())
                    .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())

                    .layer(new Convolution2D.Builder().convolutionMode(ConvolutionMode.Same).kernelSize(3, 3).stride(1, 1).nOut(32).build())
                    .layer(new BatchNormalization.Builder().build())
                    .layer(new Pooling2D.Builder().poolingType(PoolingType.MAX).stride(2, 2).build())
                    .layer(new ActivationLayer.Builder().activation(Activation.RELU).build());
            listBuilder = listBuilder
                    .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX).
                            lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).nOut(num_class).build())
                    .setInputType(InputType.convolutionalFlat(84, 84, 3));
            MultiLayerConfiguration conf = listBuilder
                    .backpropType(BackpropType.Standard)
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            INDArray features = Nd4j.randn(batch_size, 3, 84, 84); //c h w
            INDArray labels = Nd4j.create(batch_size, 1);
            for (int i = 0; i < batch_size; i++) {
                labels.putScalar(new int[]{i, 0}, random.nextInt(num_class));
            }
            long[] total_cost = new long[num_iteration];
            for (int i = 0; i < num_iteration; i++) {
                long start = new Date().getTime();
                net.fit(features, labels);
                long end = new Date().getTime();
                total_cost[i] = end - start;
            }
            System.out.println(Arrays.toString(total_cost).replaceAll(" ", ""));
        } else {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs())
                    .list()
                    .layer(new EmbeddingSequenceLayer.Builder()
                            .nIn(10000).nOut(256)
                            .build())
                    .layer(new LSTM.Builder()
                            .nIn(256).nOut(256)
                            .build())
                    .layer(new RnnOutputLayer.Builder()
                            .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT)
                            .nIn(256).nOut(10000)
                            .activation(Activation.SOFTMAX).build())
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();
            int time_step = 5;
            INDArray features = Nd4j.create(batch_size, time_step);
            INDArray labels = Nd4j.create(batch_size, 1, time_step);
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < time_step; j++) {
                    features.putScalar(new int[]{i, j}, random.nextInt(10000));
                    labels.putScalar(new int[]{i, 0, j}, random.nextInt(10000));
                }
            }
            long[] cost = new long[num_iteration];
            for (int i = 0; i < num_iteration; i++) {
                long start = new Date().getTime();
                net.fit(features, labels);
                cost[i] = new Date().getTime() - start;
            }
            System.out.println(time_step + ": " + Arrays.toString(cost));
        }
        System.out.println();
    }

    private void lookup_table_reddit(String type) throws Exception {
        Random random = new Random();
        int max_lstm_batch_size = 50, delta_lstm_batch_size = 5;
        int max_vocab_size = 10000, delta_vocab_size = 500;
        int max_size = 1000, delta_size = 50;
        int max_seq_len = 10, delta_seq_len = 2;
        int in_size, out_size, batch_size, seq_len;
        int num_iteration = 5;
        int redundant_out_size = 2;

        if (type.equals(EMBEDDING)) {
            File lookup_table_embedding = new File(EMBEDDING_DIR, "lookup_table_embedding.txt");
            if (!lookup_table_embedding.exists()) {
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_embedding));
                out.write(delta_vocab_size + "\n"); // in_size( = vocab_size) 500-20000
                out.write(delta_size + "\n");  // out_size( = embedding_size) 50-1000
                out.write(delta_lstm_batch_size + "\n");   // batch_size 5-50
                out.write(delta_seq_len + "\n");   // seq_len 5-20
                out.close();
            }

            BufferedReader reader = new BufferedReader(new FileReader(lookup_table_embedding));
            in_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            out_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            batch_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            seq_len = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            reader.close();

            while (in_size <= max_vocab_size && out_size <= max_size) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs())
                        .list()
                        .layer(new EmbeddingSequenceLayer.Builder().nIn(in_size).nOut(out_size).build())
//                        .layer(new RnnOutputLayer.Builder()
//                                .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT)
//                                .nIn(out_size).nOut(redundant_out_size)
//                                .activation(Activation.SOFTMAX).build())
                        .build();
                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();
                //train single layer network
                while (batch_size <= max_lstm_batch_size && seq_len <= max_seq_len) {
                    INDArray features = Nd4j.create(batch_size, seq_len);
                    INDArray labels = Nd4j.create(batch_size, 1, seq_len);
                    for (int i = 0; i < batch_size; i++) {
                        for (int j = 0; j < seq_len; j++) {
                            features.putScalar(new int[]{i, j}, random.nextInt(in_size));
                            labels.putScalar(new int[]{i, 0, j}, random.nextInt(redundant_out_size));
                        }
                    }
                    long[] total_cost = new long[num_iteration];
                    for (int i = 0; i < num_iteration; i++) {
                        long start = new Date().getTime();
//                        net.fit(features, labels);
                        net.output(features);
                        long end = new Date().getTime();
                        total_cost[i] = end - start;
                    }
                    String filename = EMBEDDING + "_" + in_size + "_" + out_size + "_" + batch_size + "_" + seq_len + ".txt";
                    File save_location = new File(EMBEDDING_DIR, filename);
                    BufferedWriter writer = new BufferedWriter(new FileWriter(save_location));
                    writer.write(Arrays.toString(total_cost).replaceAll(" ", ""));
                    writer.close();

                    System.out.println(in_size + ", " + out_size + ", " + batch_size + ", " + seq_len + ": " + save_location);

                    if (seq_len < max_seq_len) {
                        seq_len += delta_seq_len;
                    } else {
                        seq_len = delta_seq_len;
                        batch_size += delta_lstm_batch_size;
                    }
                    BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_embedding));
                    out.write(in_size + "\n");
                    out.write(out_size + "\n");
                    out.write(batch_size + "\n");
                    out.write(seq_len + "\n");
                    out.close();
                }

                batch_size = delta_lstm_batch_size;
                seq_len = delta_seq_len;
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_embedding));
                if (out_size < max_size) {
                    out_size += delta_size;
                } else {
                    in_size += delta_vocab_size;
                    out_size = delta_size;
                }
                out.write(in_size + "\n");
                out.write(out_size + "\n");
                out.write(batch_size + "\n");
                out.write(seq_len + "\n");
                out.close();
            }
        } else if (type.equals(LSTM)) {
            File lookup_table_lstm = new File(LSTM_DIR, "lookup_table_lstm.txt");
            if (!lookup_table_lstm.exists()) {
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_lstm));
                out.write(delta_size + "\n");  // in_size( = embedding_size) 50-1000
                out.write(delta_size + "\n");  // out_size( = hidden_size) 50-1000
                out.write(delta_lstm_batch_size + "\n");   // batch_size 5-50
                out.write(delta_seq_len + "\n");   // seq_len 5-20
                out.close();
            }

            BufferedReader reader = new BufferedReader(new FileReader(lookup_table_lstm));
            in_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            out_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            batch_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            seq_len = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            reader.close();

            while (in_size <= max_size && out_size <= max_size) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs())
                        .list()
                        .layer(new LSTM.Builder().nIn(in_size).nOut(out_size).build())
//                        .layer(new RnnOutputLayer.Builder()
//                                .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT)
//                                .nIn(out_size).nOut(redundant_out_size)
//                                .activation(Activation.SOFTMAX).build())
                        .build();
                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();
                //train single layer network
                while (batch_size <= max_lstm_batch_size && seq_len <= max_seq_len) {
                    INDArray features = Nd4j.randn(batch_size, in_size, seq_len);
                    INDArray labels = Nd4j.create(batch_size, 1, seq_len);
                    for (int i = 0; i < batch_size; i++) {
                        for (int j = 0; j < seq_len; j++) {
                            labels.putScalar(new int[]{i, 0, j}, random.nextInt(redundant_out_size));
                        }
                    }
                    long[] total_cost = new long[num_iteration];
                    for (int i = 0; i < num_iteration; i++) {
                        long start = new Date().getTime();
                        net.output(features);
//                        net.fit(features, labels);
                        long end = new Date().getTime();
                        total_cost[i] = end - start;
                    }
                    String filename = LSTM + "_" + in_size + "_" + out_size + "_" + batch_size + "_" + seq_len + ".txt";
                    File save_location = new File(LSTM_DIR, filename);
                    BufferedWriter writer = new BufferedWriter(new FileWriter(save_location));
                    writer.write(Arrays.toString(total_cost).replaceAll(" ", ""));
                    writer.close();

                    System.out.println(in_size + ", " + out_size + ", " + batch_size + ", " + seq_len + ": " + save_location);

                    if (seq_len < max_seq_len) {
                        seq_len += delta_seq_len;
                    } else {
                        seq_len = delta_seq_len;
                        batch_size += delta_lstm_batch_size;
                    }
                    BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_lstm));
                    out.write(in_size + "\n");
                    out.write(out_size + "\n");
                    out.write(batch_size + "\n");
                    out.write(seq_len + "\n");
                    out.close();
                }

                seq_len = delta_seq_len;
                batch_size = delta_lstm_batch_size;
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_lstm));
                if (out_size < max_size) {
                    out_size += delta_size;
                } else {
                    in_size += delta_size;
                    out_size = delta_size;
                }
                out.write(in_size + "\n");
                out.write(out_size + "\n");
                out.write(batch_size + "\n");
                out.write(seq_len + "\n");
                out.close();
            }
        } else if (type.equals(RNN_OUTPUT)) {
            File lookup_table_rnn_output = new File(RNN_OUTPUT_DIR, "lookup_table_rnn_output.txt");
            if (!lookup_table_rnn_output.exists()) {
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_rnn_output));
                out.write(delta_size + "\n");  // in_size( = hidden_size) 50-1000
                out.write(delta_vocab_size + "\n");  // out_size( = vocab_size) 500-20000
                out.write(delta_lstm_batch_size + "\n");   // batch_size 5-50
                out.write(delta_seq_len + "\n");   // seq_len 5-20
                out.close();
            }

            BufferedReader reader = new BufferedReader(new FileReader(lookup_table_rnn_output));
            in_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            out_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            batch_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            seq_len = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            reader.close();

            if (in_size <= max_size && out_size <= max_vocab_size) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs())
                        .list()
                        .layer(new RnnOutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT)
                                .nIn(in_size).nOut(out_size)
                                .activation(Activation.SOFTMAX).build())
                        .build();
                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();
                //train single layer network
                while (batch_size <= max_lstm_batch_size && seq_len <= max_seq_len) {
                    INDArray features = Nd4j.randn(batch_size, in_size, seq_len);
                    INDArray labels = Nd4j.create(batch_size, 1, seq_len);
                    for (int i = 0; i < batch_size; i++) {
                        for (int j = 0; j < seq_len; j++) {
                            labels.putScalar(new int[]{i, 0, j}, random.nextInt(out_size));
                        }
                    }
                    long[] total_cost = new long[num_iteration];
                    for (int i = 0; i < num_iteration; i++) {
                        long start = new Date().getTime();
                        net.output(features);
//                        net.fit(features, labels);
                        long end = new Date().getTime();
                        total_cost[i] = end - start;
                    }
                    String filename = RNN_OUTPUT + "_" + in_size + "_" + out_size + "_" + batch_size + "_" + seq_len + ".txt";
                    File save_location = new File(RNN_OUTPUT_DIR, filename);
                    BufferedWriter writer = new BufferedWriter(new FileWriter(save_location));
                    writer.write(Arrays.toString(total_cost).replaceAll(" ", ""));
                    writer.close();

                    System.out.println(in_size + ", " + out_size + ", " + batch_size + ", " + seq_len + ": " + save_location);

                    if (seq_len < max_seq_len) {
                        seq_len += delta_seq_len;
                    } else {
                        seq_len = delta_seq_len;
                        batch_size += delta_lstm_batch_size;
                    }
                    BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_rnn_output));
                    out.write(in_size + "\n");
                    out.write(out_size + "\n");
                    out.write(batch_size + "\n");
                    out.write(seq_len + "\n");
                    out.close();
                }
                System.gc();

                batch_size = delta_lstm_batch_size;
                seq_len = delta_seq_len;
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_rnn_output));
                if (out_size < max_vocab_size) {
                    out_size += delta_vocab_size;
                } else {
                    in_size += delta_size;
                    out_size = delta_vocab_size;

                }
                out.write(in_size + "\n");
                out.write(out_size + "\n");
                out.write(batch_size + "\n");
                out.write(seq_len + "\n");
                out.close();
            }
        }
    }

    private void lookup_table_celeba(String type) throws Exception {
        Random random = new Random();
        int num_iteration = 1, redundent_out_size = 2;
        long[] total_cost = new long[num_iteration];
        //conv_2d
        int max_kernel_size = 9, delta_kernel_size = 2, min_kernel_size = 1;
        int max_stride = 5, delta_stride = 1, min_stride = 1;
        int max_channel = 10, delta_channel = 1, min_channel = 1;  // in_size
        int max_filter_num = 10, delta_filter_num = 1, min_filter_num = 1; //out_size
        int max_image_size = 100, delta_image_size = 10, min_image_size = 10;
        int max_cnn_batch_size = 10, delta_cnn_batch_size = 1, min_cnn_batch_size = 1;
        //output
        int max_in_size = 1000, delta_in_size = 50;
        int max_out_size = 100, delta_out_size = 2;
        //params
        int kernel_size, stride, filter_num, image_size, channel, batch_size;
        int in_size, out_size;
        if (type.equals(CONVOLUTION_2D)) {
            File lookup_table_con2d = new File(CONVOLUTION_2D_DIR, "lookup_table_con2d.txt");
            if (!lookup_table_con2d.exists()) {
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_con2d));
                out.write(min_kernel_size + "\n");
                out.write(min_stride + "\n");
                out.write(min_channel + "\n");
                out.write(min_filter_num + "\n");
                out.write(min_image_size + "\n");
                out.write(min_cnn_batch_size + "\n");
                out.close();
            }

            BufferedReader reader = new BufferedReader(new FileReader(lookup_table_con2d));
            kernel_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            stride = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            channel = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            filter_num = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            image_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            batch_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            reader.close();

            while (kernel_size <= max_kernel_size && stride <= max_stride && channel <= max_channel &&
                    filter_num <= max_filter_num && image_size <= max_image_size) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new AdaGrad(0.1))
                        .list()
                        .layer(new Convolution2D.Builder().kernelSize(kernel_size, kernel_size).stride(stride, stride)
                                .nOut(filter_num).build())
                        .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).nOut(redundent_out_size).build())
                        .setInputType(InputType.convolutional(image_size, image_size, channel))
                        .backpropType(BackpropType.Standard)
                        .build();
                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();
                while (batch_size <= max_cnn_batch_size) {
                    INDArray features = Nd4j.rand(batch_size, channel, image_size, image_size);
                    INDArray labels = Nd4j.zeros(batch_size, 1);
                    for (int i = 0; i < batch_size; i++) {
                        labels.putScalar(new int[]{i, 0}, random.nextInt(redundent_out_size));
                    }
                    for (int i = 0; i < num_iteration; i++) {
                        long start = new Date().getTime();
                        net.fit(features, labels);
                        total_cost[i] = new Date().getTime() - start;
                    }
                    String filename = CONVOLUTION_2D +
                            "_" + kernel_size +
                            "_" + stride +
                            "_" + channel +
                            "_" + filter_num +
                            "_" + image_size +
                            "_" + batch_size + ".txt";
                    File save_location = new File(CONVOLUTION_2D_DIR, filename);
                    BufferedWriter writer = new BufferedWriter(new FileWriter(save_location));
                    writer.write(Arrays.toString(total_cost));
                    writer.close();
                    System.out.println(save_location);

                    batch_size += delta_cnn_batch_size;
                    BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_con2d));
                    out.write(kernel_size + "\n");
                    out.write(stride + "\n");
                    out.write(channel + "\n");
                    out.write(filter_num + "\n");
                    out.write(image_size + "\n");
                    out.write(batch_size + "\n");
                    out.close();
                }
                batch_size = min_cnn_batch_size;
                if (image_size < max_image_size) {
                    image_size += delta_image_size;
                } else if (filter_num < max_filter_num) {
                    filter_num += delta_filter_num;
                    image_size = min_image_size;
                } else if (channel < max_channel) {
                    channel += delta_channel;
                    filter_num = min_filter_num;
                    image_size = min_image_size;
                } else if (stride < max_stride) {
                    stride += delta_stride;
                    channel = min_channel;
                    filter_num = min_filter_num;
                    image_size = min_image_size;
                } else {
                    kernel_size += delta_kernel_size;
                    stride = min_stride;
                    channel = min_channel;
                    filter_num = min_filter_num;
                    image_size = min_image_size;
                }
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_con2d));
                out.write(kernel_size + "\n");
                out.write(stride + "\n");
                out.write(channel + "\n");
                out.write(filter_num + "\n");
                out.write(image_size + "\n");
                out.write(batch_size + "\n");
                out.close();
            }

        } else if (type.equals(OUTPUT)) {
            File lookup_table_output = new File(OUTPUT_DIR, "lookup_table_output.txt");
            if (!lookup_table_output.exists()) {
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_output));
                out.write(delta_in_size + "\n");
                out.write(delta_out_size + "\n");
                out.write(delta_cnn_batch_size + "\n");
                out.close();
            }

            BufferedReader reader = new BufferedReader(new FileReader(lookup_table_output));
            in_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            out_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            batch_size = Integer.parseInt(reader.readLine().replaceAll("[\\n\\r]", ""));
            reader.close();

            while (in_size <= max_in_size && out_size <= max_out_size) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new AdaGrad(0.1))
                        .list()
                        .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.SPARSE_MCXENT).nIn(in_size).nOut(out_size).build())
                        .backpropType(BackpropType.Standard)
                        .build();
                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();
                while (batch_size <= max_cnn_batch_size) {
                    INDArray features = Nd4j.rand(batch_size, in_size);
                    INDArray labels = Nd4j.zeros(batch_size, 1);
                    for (int i = 0; i < batch_size; i++) {
                        labels.putScalar(new int[]{i, 0}, random.nextInt(out_size));
                    }
                    for (int i = 0; i < num_iteration; i++) {
                        long start = new Date().getTime();
                        net.fit(features, labels);
                        total_cost[i] = new Date().getTime() - start;
                    }
                    String filename = OUTPUT +
                            "_" + in_size +
                            "_" + out_size +
                            "_" + batch_size + ".txt";
                    File save_location = new File(OUTPUT_DIR, filename);
                    BufferedWriter writer = new BufferedWriter(new FileWriter(save_location));
                    writer.write(Arrays.toString(total_cost));
                    writer.close();
                    System.out.println(save_location);

                    batch_size += delta_cnn_batch_size;
                    BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_output));
                    out.write(in_size + "\n");
                    out.write(out_size + "\n");
                    out.write(batch_size + "\n");
                    out.close();
                }

                batch_size = delta_cnn_batch_size;
                if (out_size < max_out_size) {
                    out_size += delta_out_size;
                } else {
                    in_size += delta_in_size;
                    out_size = delta_out_size;
                }
                BufferedWriter out = new BufferedWriter(new FileWriter(lookup_table_output));
                out.write(in_size + "\n");
                out.write(out_size + "\n");
                out.write(batch_size + "\n");
                out.close();
            }
        }
    }

    private void multiRegresssion() {
        int seed = 12345;
        int nSamples = 1000;
        int batchSize = 100;
        double learningRate = 0.01;
        Random rng = new Random(seed);
        int numInputs = 1;
        int numOutputs = 1;
        int numHiddenNodes = 100;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .backpropType(BackpropType.Standard)
                .build();

        //Generate the training data
        INDArray x = Nd4j.linspace(-10, 10, nSamples).reshape(nSamples, 1);
        INDArray y = Nd4j.getExecutioner().execAndReturn(new Sin(x.dup())).z().div(x);
        DataSet allData = new DataSet(x, y);

        List<DataSet> list = allData.asList();
        Collections.shuffle(list, rng);
        DataSetIterator iterator = new ListDataSetIterator(list, batchSize);

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        long start = new Date().getTime();
        net.fit(iterator, 1);
        long cost = new Date().getTime() - start;
        SaveLoadMultiLayerNetwork(net, "multiRegresssion", cost);
    }

    private void SaveLoadComputationGraph(ComputationGraph net, String type, Long cost) {
        String filename = "CG_" + type + "_" + new Date().getTime() + "_" + cost + ".zip";

        File locationToSave = new File(getFilesDir(), filename);
        try {
            ModelSerializer.writeModel(net, locationToSave, true);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Load the model
//        try {
//            ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
//            System.out.println("Saved and loaded parameters are equal: " +
//                    net.params().equals(restored.params()));
//            System.out.println("Saved and loaded configurations are equal: " +
//                    net.getConfiguration().equals(restored.getConfiguration()));
//        } catch (Exception e) {
//            System.out.println("");
//        }
    }

    private void SaveLoadMultiLayerNetwork(MultiLayerNetwork net, String type, Long cost) {
        String filename = "MLN_" + type + "_" + new Date().getTime() + "_" + cost + ".zip";

        File locationToSave = new File(getFilesDir(), filename);
        System.out.println(locationToSave);
        if (!locationToSave.exists()) {
            try {
                locationToSave.createNewFile();
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
        try {
            ModelSerializer.writeModel(net, locationToSave, true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
