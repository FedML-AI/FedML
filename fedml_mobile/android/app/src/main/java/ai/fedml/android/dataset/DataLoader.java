package ai.fedml.android.dataset;

import android.os.Environment;

import java.io.File;

public class DataLoader {
    // entire model
    private String
            CELEBA = "celeba",
            REDDIT = "reddit",
            FEMNIST = "femnist",
            SHAKESPEARE = "shakespeare",
            SENT140 = "sent140", SENT140_RNN = "sent140_stacked_lstm", SENT140_REG = "sent140_bag_log_reg";

    private File IRIS_DIR = new File(Environment.getExternalStorageDirectory(), "iris_classifier");

    private File REDDIT_DIR = new File(IRIS_DIR, REDDIT);
    private File CELEBA_DIR = new File(IRIS_DIR, CELEBA);
    private File FEMNIST_DIR = new File(IRIS_DIR, FEMNIST);
    private File SHAKESPEARE_DIR = new File(IRIS_DIR, SHAKESPEARE);
    private File SENT140_DIR = new File(IRIS_DIR, SENT140),
            SENT140_RNN_DIR = new File(SENT140_DIR, SENT140_RNN),
            SENT140_REG_DIR = new File(SENT140_DIR, SENT140_REG);


    // reddit model layers
    private String EMBEDDING = "embedding", LSTM = "lstm", RNN_OUTPUT = "rnn_output";

    //celeba model layers
    private String CONVOLUTION_2D = "convolution_2d", OUTPUT = "output";
    private String BATCH_NORMALIZATION = "batch_normalization", POOLING_2D = "pooling_2d";

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

}
