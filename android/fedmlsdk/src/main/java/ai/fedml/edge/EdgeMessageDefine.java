package ai.fedml.edge;

public interface EdgeMessageDefine {
    int MSG_FROM_CLIENT = 1;
    int MSG_FROM_SERVICE = 2;
    int MSG_START_TRAIN = 3;
    int MSG_TRAIN_STATUS = 4;
    int MSG_TRAIN_PROGRESS = 5;
    int MSG_BIND_EDGE = 6;
    int MSG_TRAIN_ACCURACY = 7;
    int MSG_TRAIN_LOSS = 8;

    String TRAIN_ARGS = "train_args";
    String TRAIN_EPOCH = "train_epoch";
    String TRAIN_LOSS = "train_loss";
    String TRAIN_ACCURACY = "train_accuracy";
    String BIND_EDGE_ID = "bind_id";

    int IDLE = 0;
    int INITIALIZING = 1;
    int TRAINING = 2;
    int STOPPING = 3;
    int FINISHED = 4;
    int UPGRADING = 5;
}
